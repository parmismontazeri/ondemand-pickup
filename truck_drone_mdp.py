import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
from enum import Enum
import heapq

class VehicleType(Enum):
    TRUCK = "truck"
    DRONE = "drone"

class CustomerType(Enum):
    SCHEDULED_DELIVERY = "scheduled_delivery"
    ONDEMAND_PICKUP = "ondemand_pickup"

class DroneStatus(Enum):
    ON_TRUCK = "on_truck"
    IN_FLIGHT = "in_flight"
    WAITING = "waiting"

@dataclass
class Customer:
    id: int
    location: Tuple[float, float]
    customer_type: CustomerType
    demand: float
    time_window: Tuple[float, float]  # (earliest, latest)
    deadline: Optional[float] = None
    revenue: float = 0.0
    service_time: float = 0.0
    revealed: bool = False  # For on-demand customers
    
@dataclass
class Vehicle:
    id: int
    vehicle_type: VehicleType
    location: Tuple[float, float]
    capacity: float
    current_load: float = 0.0
    speed: float = 1.0
    
@dataclass
class Drone(Vehicle):
    battery_capacity: float = 100.0
    current_battery: float = 100.0
    energy_consumption_rate: float = 1.0
    status: DroneStatus = DroneStatus.ON_TRUCK
    launch_location: Optional[Tuple[float, float]] = None
    retrieval_location: Optional[Tuple[float, float]] = None
    
@dataclass
class State:
    """MDP State representation"""
    current_time: float
    truck_location: Tuple[float, float]
    drone_location: Tuple[float, float]
    drone_status: DroneStatus
    drone_battery: float
    truck_load: float
    drone_load: float
    unserved_customers: Set[int]
    served_customers: Set[int]
    current_route: List[int]
    waiting_at_location: bool = False
    revealed_ondemand_customers: Set[int] = field(default_factory=set)
    
@dataclass
class Action:
    """MDP Action representation"""
    action_type: str  # 'truck_serve', 'drone_launch', 'drone_retrieve', 'wait', 'end'
    target_customer: Optional[int] = None
    drone_customers: List[int] = field(default_factory=list)  # For multi-visit drone trips
    next_truck_location: Optional[int] = None

class TruckDroneMDP:
    """
    MDP formulation for Dynamic Truck-Drone Routing Problem
    with Scheduled Deliveries and On-demand Pickups
    """
    
    def __init__(self, 
                 depot: Tuple[float, float],
                 scheduled_customers: List[Customer],
                 truck_capacity: float = 100.0,
                 drone_capacity: float = 10.0,
                 drone_battery: float = 100.0,
                 truck_speed: float = 1.0,
                 drone_speed: float = 2.0,
                 working_hours: float = 8.0,
                 truck_cost_per_time: float = 1.0,
                 drone_cost_per_time: float = 0.5,
                 buffer_time: float = 0.5,
                 penalty_cost: float = 10.0):
        
        self.depot = depot
        self.scheduled_customers = scheduled_customers
        self.ondemand_customers = []  # Dynamically revealed
        self.all_customers = {c.id: c for c in scheduled_customers}
        
        # Vehicle parameters
        self.truck_capacity = truck_capacity
        self.drone_capacity = drone_capacity
        self.drone_battery = drone_battery
        self.truck_speed = truck_speed
        self.drone_speed = drone_speed
        # Buffer time reserved at each truck stop to create a "loose" plan
        # which gives time slack for accepting on-demand pickups.
        self.buffer_time = buffer_time
        
        # Problem parameters
        self.working_hours = working_hours
        self.truck_cost_per_time = truck_cost_per_time
        self.drone_cost_per_time = drone_cost_per_time
        self.penalty_cost = penalty_cost
        
        # State tracking
        self.current_state: Optional[State] = None
        self.total_profit = 0.0
        
    def initialize_state(self) -> State:
        """Initialize the MDP state at the beginning"""
        unserved = {c.id for c in self.scheduled_customers}
        
        return State(
            current_time=0.0,
            truck_location=self.depot,
            drone_location=self.depot,
            drone_status=DroneStatus.ON_TRUCK,
            drone_battery=self.drone_battery,
            truck_load=0.0,
            drone_load=0.0,
            unserved_customers=unserved,
            served_customers=set(),
            current_route=[],
            waiting_at_location=False,
            revealed_ondemand_customers=set()
        )
    
    def get_feasible_actions(self, state: State) -> List[Action]:
        """Generate all feasible actions from current state"""
        actions = []
        
        # 1. Truck serves customer directly
        for cid in state.unserved_customers:
            customer = self.all_customers[cid]
            if self._is_truck_service_feasible(state, customer):
                actions.append(Action(
                    action_type='truck_serve',
                    target_customer=cid
                ))
        
        # 2. Launch drone for multi-visit trip
        if state.drone_status == DroneStatus.ON_TRUCK:
            drone_trips = self._generate_drone_trips(state)
            for trip in drone_trips:
                actions.append(Action(
                    action_type='drone_launch',
                    drone_customers=trip['customers'],
                    next_truck_location=trip['retrieval_node']
                ))
        
        # 3. Retrieve drone
        if state.drone_status == DroneStatus.WAITING:
            actions.append(Action(action_type='drone_retrieve'))
        
        # 4. Wait for drone synchronization
        if state.drone_status == DroneStatus.IN_FLIGHT:
            actions.append(Action(action_type='wait'))
        
        # 5. End operations (return to depot)
        if len(state.unserved_customers) == 0:
            actions.append(Action(action_type='end'))
        
        return actions
    
    def _is_truck_service_feasible(self, state: State, customer: Customer) -> bool:
        """Check if truck can feasibly serve a customer"""
        # Check capacity
        if state.truck_load + customer.demand > self.truck_capacity:
            return False
        
        # Check time feasibility
        travel_time = self._calculate_distance(state.truck_location, customer.location) / self.truck_speed
        arrival_time = state.current_time + travel_time
        
        # Check deadline
        if customer.deadline and arrival_time > customer.deadline:
            return False
        
        # Check working hours
        if arrival_time > self.working_hours:
            return False
        
        return True
    
    def _generate_drone_trips(self, state: State) -> List[Dict]:
        """Generate feasible multi-visit drone trips"""
        trips = []
        unserved = list(state.unserved_customers)
        
        # Simple heuristic: generate trips with 1-3 customers
        for i in range(len(unserved)):
            for j in range(i, min(i+3, len(unserved))):
                trip_customers = unserved[i:j+1]
                
                if self._is_drone_trip_feasible(state, trip_customers):
                    # Determine retrieval node
                    retrieval_node = self._select_retrieval_node(state, trip_customers)
                    trips.append({
                        'customers': trip_customers,
                        'retrieval_node': retrieval_node
                    })
        
        return trips
    
    def _is_drone_trip_feasible(self, state: State, customer_ids: List[int]) -> bool:
        """Check if drone trip is feasible"""
        total_demand = sum(self.all_customers[cid].demand for cid in customer_ids)
        
        # Check capacity
        if total_demand > self.drone_capacity:
            return False
        
        # Check battery/endurance
        total_distance = self._calculate_drone_trip_distance(state.truck_location, customer_ids)
        energy_needed = total_distance * self.drone_speed  # Simplified energy model
        
        if energy_needed > state.drone_battery:
            return False
        
        return True
    
    def _calculate_drone_trip_distance(self, start: Tuple[float, float], 
                                      customer_ids: List[int]) -> float:
        """Calculate total distance for drone trip"""
        total_dist = 0.0
        current_loc = start
        
        for cid in customer_ids:
            customer_loc = self.all_customers[cid].location
            total_dist += self._calculate_distance(current_loc, customer_loc)
            current_loc = customer_loc
        
        # Return to truck (simplified - assumes truck waits)
        total_dist += self._calculate_distance(current_loc, start)
        return total_dist
    
    def _select_retrieval_node(self, state: State, customer_ids: List[int]) -> int:
        """Select retrieval node for drone (simplified)"""
        # Simple heuristic: return the last customer in the trip
        return customer_ids[-1] if customer_ids else None
    
    def transition(self, state: State, action: Action) -> Tuple[State, float]:
        """
        Execute action and return next state and reward
        
        Returns:
            next_state: The resulting state
            reward: Immediate reward for this transition
        """
        new_state = self._copy_state(state)
        reward = 0.0
        
        if action.action_type == 'truck_serve':
            new_state, reward = self._execute_truck_serve(new_state, action)
            
        elif action.action_type == 'drone_launch':
            new_state, reward = self._execute_drone_launch(new_state, action)
            
        elif action.action_type == 'drone_retrieve':
            new_state, reward = self._execute_drone_retrieve(new_state, action)
            
        elif action.action_type == 'wait':
            new_state, reward = self._execute_wait(new_state, action)
            
        elif action.action_type == 'end':
            new_state, reward = self._execute_end(new_state, action)
        
        # Check for newly revealed on-demand customers
        new_state = self._reveal_ondemand_customers(new_state)
        
        return new_state, reward
    
    def _execute_truck_serve(self, state: State, action: Action) -> Tuple[State, float]:
        """Execute truck serving a customer"""
        customer = self.all_customers[action.target_customer]
        
        # Calculate travel time and costs
        travel_time = self._calculate_distance(state.truck_location, customer.location) / self.truck_speed
        service_time = customer.service_time
        
        # Update state
        state.current_time += travel_time + service_time
        state.truck_location = customer.location
        state.truck_load += customer.demand
        state.unserved_customers.discard(action.target_customer)
        state.served_customers.add(action.target_customer)
        state.current_route.append(action.target_customer)
        
        # Calculate reward
        reward = customer.revenue - (travel_time + service_time) * self.truck_cost_per_time
        
        # Apply penalty if late
        if customer.deadline and state.current_time > customer.deadline:
            reward -= self.penalty_cost
        
        return state, reward
    
    def _execute_drone_launch(self, state: State, action: Action) -> Tuple[State, float]:
        """Execute drone launch for multi-visit trip"""
        # Calculate flight time
        flight_time = self._calculate_drone_trip_distance(
            state.truck_location, action.drone_customers
        ) / self.drone_speed
        
        # Update state
        state.drone_status = DroneStatus.IN_FLIGHT
        retrieval_location = self.all_customers[action.next_truck_location].location
        state.drone_location = retrieval_location  # Simplified
        
        # Calculate rewards from served customers
        reward = 0.0
        for cid in action.drone_customers:
            customer = self.all_customers[cid]
            reward += customer.revenue
            state.unserved_customers.discard(cid)
            state.served_customers.add(cid)
        
        # Subtract operation costs
        reward -= flight_time * self.drone_cost_per_time
        
        return state, reward
    
    def _execute_drone_retrieve(self, state: State, action: Action) -> Tuple[State, float]:
        """Execute drone retrieval"""
        state.drone_status = DroneStatus.ON_TRUCK
        state.drone_location = state.truck_location
        state.drone_battery = self.drone_battery  # Recharge
        return state, 0.0
    
    def _execute_wait(self, state: State, action: Action) -> Tuple[State, float]:
        """Execute wait for synchronization"""
        # Simple wait - advance time slightly
        state.current_time += 0.1
        state.waiting_at_location = True
        return state, 0.0
    
    def _execute_end(self, state: State, action: Action) -> Tuple[State, float]:
        """Return to depot and end"""
        travel_time = self._calculate_distance(state.truck_location, self.depot) / self.truck_speed
        state.current_time += travel_time
        state.truck_location = self.depot
        reward = -travel_time * self.truck_cost_per_time
        return state, reward
    
    def _reveal_ondemand_customers(self, state: State) -> State:
        """Simulate revelation of on-demand pickup requests"""
        # Simplified: randomly reveal customers based on time
        # In practice, this would be based on actual customer requests
        return state
    
    def _copy_state(self, state: State) -> State:
        """Deep copy state"""
        return State(
            current_time=state.current_time,
            truck_location=state.truck_location,
            drone_location=state.drone_location,
            drone_status=state.drone_status,
            drone_battery=state.drone_battery,
            truck_load=state.truck_load,
            drone_load=state.drone_load,
            unserved_customers=state.unserved_customers.copy(),
            served_customers=state.served_customers.copy(),
            current_route=state.current_route.copy(),
            waiting_at_location=state.waiting_at_location,
            revealed_ondemand_customers=state.revealed_ondemand_customers.copy()
        )
    
    @staticmethod
    def _calculate_distance(loc1: Tuple[float, float], loc2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two locations"""
        return np.sqrt((loc1[0] - loc2[0])**2 + (loc1[1] - loc2[1])**2)
    
    def is_terminal_state(self, state: State) -> bool:
        """Check if state is terminal"""
        return (len(state.unserved_customers) == 0 and 
                state.drone_status == DroneStatus.ON_TRUCK)


# Example usage and testing
if __name__ == "__main__":
    # Create example problem instance
    depot = (0.0, 0.0)
    
    customers = [
        Customer(id=1, location=(10, 10), customer_type=CustomerType.SCHEDULED_DELIVERY,
                demand=5, time_window=(0, 100), revenue=50, service_time=2),
        Customer(id=2, location=(20, 5), customer_type=CustomerType.SCHEDULED_DELIVERY,
                demand=3, time_window=(0, 100), revenue=30, service_time=2),
        Customer(id=3, location=(15, 20), customer_type=CustomerType.SCHEDULED_DELIVERY,
                demand=4, time_window=(0, 100), revenue=40, service_time=2),
        Customer(id=4, location=(5, 25), customer_type=CustomerType.SCHEDULED_DELIVERY,
                demand=2, time_window=(0, 100), revenue=25, service_time=2),
    ]
    
    # Initialize MDP
    mdp = TruckDroneMDP(
        depot=depot,
        scheduled_customers=customers,
        truck_capacity=20,
        drone_capacity=8,
        drone_battery=100,
        truck_speed=1.0,
        drone_speed=2.0,
        working_hours=100
    )
    
    # Initialize state
    state = mdp.initialize_state()
    print(f"Initial state: {len(state.unserved_customers)} unserved customers")
    print(f"Truck at: {state.truck_location}")
    print(f"Drone status: {state.drone_status}")
    
    # Get feasible actions
    actions = mdp.get_feasible_actions(state)
    print(f"\nFeasible actions: {len(actions)}")
    for i, action in enumerate(actions[:5]):  # Show first 5
        print(f"  Action {i}: {action.action_type}", end="")
        if action.target_customer:
            print(f" -> Customer {action.target_customer}")
        elif action.drone_customers:
            print(f" -> Drone trip: {action.drone_customers}")
        else:
            print()
    
    # Execute a sample action
    if actions:
        action = actions[0]
        next_state, reward = mdp.transition(state, action)
        print(f"\nAfter action: {action.action_type}")
        print(f"  Reward: {reward:.2f}")
        print(f"  Time: {next_state.current_time:.2f}")
        print(f"  Unserved customers: {len(next_state.unserved_customers)}")
        print(f"  Truck location: {next_state.truck_location}")
