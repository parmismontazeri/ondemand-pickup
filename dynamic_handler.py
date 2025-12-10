"""
Dynamic Request Handler
Processes on-demand delivery requests during operations.

Evaluates feasibility, generates insertion options, and updates the operational plan
in real-time as new customer requests arrive.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional


class DynamicRequestHandler:
    """
    Handles dynamic customer requests arriving during operations.
    
    For each request, evaluates:
    1. Is the request feasible (time window, capacity)?
    2. What are the truck/drone insertion options?
    3. Which option maximizes profit?
    4. Update the operational plan accordingly
    
    This enables real-time decision-making as demand evolves.
    """
    
    def __init__(self, mdp, initial_state, truck_route, drone_trips):
        """
        Initialize request handler with current operational state
        
        Args:
            mdp: TruckDroneMDP instance
            initial_state: Current MDP state
            truck_route: Current truck route (list of customer IDs)
            drone_trips: Current drone trips (list of dicts)
        """
        self.mdp = mdp
        self.state = initial_state
        self.truck_route = truck_route.copy()
        self.drone_trips = [trip.copy() for trip in drone_trips]
        
        self.all_customers = mdp.all_customers
        self.depot = mdp.depot
        
        self.truck_capacity = mdp.truck_capacity
        self.drone_capacity = mdp.drone_capacity
        self.truck_speed = mdp.truck_speed
        self.drone_speed = mdp.drone_speed
        self.drone_battery = mdp.drone_battery
        
        self.truck_cost_per_time = mdp.truck_cost_per_time
        self.drone_cost_per_time = mdp.drone_cost_per_time
        
        # Statistics
        self.requests_processed = 0
        self.requests_accepted = 0
        self.requests_rejected = 0
        self.total_accepted_profit = 0.0
        self.accepted_request_ids = []
    
    def handle_request(self, request_customer) -> Dict:
        """
        Process a single dynamic customer request
        
        Args:
            request_customer: Customer object with delivery request
            
        Returns:
            Dictionary with decision and metrics
        """
        self.requests_processed += 1
        request_id = request_customer.id
        
        print(f"\n[Request {self.requests_processed}] Processing request from customer {request_id}")
        print(f"  Demand: {request_customer.demand}, Revenue: ${request_customer.revenue:.2f}")
        print(f"  Location: {request_customer.location}, Deadline: {request_customer.deadline}")
        
        # CHECK 1: Basic feasibility
        if not self._is_request_feasible(request_customer):
            print(f"  X Request rejected: Not feasible")
            self.requests_rejected += 1
            return {
                'accepted': False,
                'request_id': request_id,
                'reason': 'Not feasible',
                'profit': 0.0
            }
        
        # CHECK 2: Find insertion options
        truck_option = self._find_truck_insertion(request_customer)
        drone_option = self._find_drone_insertion(request_customer)
        
        # CHECK 3: Evaluate best option
        best_option = None
        best_profit = -float('inf')
        best_type = None
        
        if truck_option:
            profit = self._evaluate_insertion(truck_option, 'truck')
            print(f"  Truck option: insertion cost ${truck_option['insertion_cost']:.2f}, "
                  f"profit ${profit:.2f}")
            if profit > best_profit:
                best_profit = profit
                best_option = truck_option
                best_type = 'truck'
        
        if drone_option:
            profit = self._evaluate_insertion(drone_option, 'drone')
            print(f"  Drone option: insertion cost ${drone_option['insertion_cost']:.2f}, "
                  f"profit ${profit:.2f}")
            if profit > best_profit:
                best_profit = profit
                best_option = drone_option
                best_type = 'drone'
        
        # CHECK 4: Accept if profitable
        if best_option and best_profit >= 0:
            # Execute insertion
            if best_type == 'truck':
                self._execute_truck_insertion(request_customer, best_option)
            else:  # drone
                self._execute_drone_insertion(request_customer, best_option)
            
            print(f"  OK Request accepted via {best_type.upper()}: profit ${best_profit:.2f}")
            self.requests_accepted += 1
            self.total_accepted_profit += best_profit
            self.accepted_request_ids.append(request_id)
            
            return {
                'accepted': True,
                'request_id': request_id,
                'method': best_type,
                'profit': best_profit,
                'insertion_cost': best_option['insertion_cost']
            }
        else:
            print(f"  X Request rejected: Not profitable")
            self.requests_rejected += 1
            return {
                'accepted': False,
                'request_id': request_id,
                'reason': 'Not profitable',
                'profit': 0.0
            }
    
    # ========================================================================
    # STEP 1: FEASIBILITY CHECK
    # ========================================================================
    
    def _is_request_feasible(self, customer) -> bool:
        """
        Check if request can possibly be served
        
        Args:
            customer: Request customer
            
        Returns:
            True if request is feasible
        """
        # Must fit in either vehicle
        if customer.demand > self.truck_capacity and customer.demand > self.drone_capacity:
            print(f"    X Demand {customer.demand} exceeds both vehicle capacities")
            return False
        
        # Must have time window feasibility (simplified check)
        current_time = self.state.current_time if hasattr(self.state, 'current_time') else 0
        if customer.deadline <= current_time:
            print(f"    X Deadline {customer.deadline} already passed")
            return False
        
        return True
    
    # ========================================================================
    # STEP 2: FIND INSERTION OPTIONS
    # ========================================================================
    
    def _find_truck_insertion(self, customer) -> Optional[Dict]:
        """
        Find all feasible insertion points in truck route
        
        Args:
            customer: Request customer
            
        Returns:
            Best truck insertion option, or None if not feasible
        """
        if customer.demand > self.truck_capacity:
            return None  # Exceeds truck capacity
        
        # Check current truck load
        truck_load = sum(self.all_customers[cid].demand for cid in self.truck_route)
        if truck_load + customer.demand > self.truck_capacity:
            return None  # Would exceed capacity
        
        # Find best insertion position
        best_insertion = None
        min_cost_increase = float('inf')
        
        # Try inserting at each position
        for i in range(len(self.truck_route) + 1):
            # Calculate insertion cost
            cost_increase = self._calculate_truck_insertion_cost(customer, i)
            
            if cost_increase < min_cost_increase:
                min_cost_increase = cost_increase
                best_insertion = {
                    'position': i,
                    'insertion_cost': cost_increase,
                    'customer_id': customer.id,
                    'method': 'truck'
                }
        
        return best_insertion
    
    def _find_drone_insertion(self, customer) -> Optional[Dict]:
        """
        Find feasible drone insertion option
        
        Args:
            customer: Request customer
            
        Returns:
            Drone insertion option, or None if not feasible
        """
        if customer.demand > self.drone_capacity:
            return None  # Exceeds drone capacity

        # Try inserting as a drone trip between any two consecutive truck nodes
        best_option = None
        min_cost = float('inf')
        # Precompute truck arrival times
        truck_times = self._compute_truck_arrival_times(self.truck_route)

        # Consider launch position i (0..len) where launch is previous node (or depot)
        for i in range(len(self.truck_route) + 1):
            if i == 0:
                launch_node = 0
            else:
                launch_node = self.truck_route[i - 1]

            if i == len(self.truck_route):
                retrieval_node = 0
            else:
                retrieval_node = self.truck_route[i]

            # Check battery feasibility for single-customer trip
            if not self._is_trip_battery_feasible([customer.id]):
                continue

            # Compute drone insertion cost (operation cost)
            drone_cost = self._calculate_drone_insertion_cost([customer.id], launch_node, retrieval_node)

            # Compute truck gap and waiting time
            t_launch = truck_times.get(launch_node, truck_times.get(0, 0.0))
            t_retrieval = truck_times.get(retrieval_node, truck_times.get(0 + 0.1, 0.0))
            drone_duration = self._drone_trip_duration(launch_node, [customer.id], retrieval_node)
            truck_gap = max(0.0, t_retrieval - t_launch)
            waiting_time = max(0.0, drone_duration - truck_gap)
            waiting_cost = waiting_time * self.truck_cost_per_time

            insertion_cost = drone_cost + waiting_cost

            if insertion_cost < min_cost:
                min_cost = insertion_cost
                best_option = {
                    'trip_customers': [customer.id],
                    'insertion_cost': insertion_cost,
                    'customer_id': customer.id,
                    'method': 'drone',
                    'launch_node': launch_node,
                    'retrieval_node': retrieval_node
                }

        return best_option
    
    # ========================================================================
    # STEP 3: EVALUATE INSERTION OPTIONS
    # ========================================================================
    
    def _evaluate_insertion(self, option: Dict, method: str) -> float:
        """
        Calculate profit of accepting this insertion
        
        Args:
            option: Insertion option (truck or drone)
            method: 'truck' or 'drone'
            
        Returns:
            Net profit (revenue - insertion cost)
        """
        customer = self.all_customers[option['customer_id']]
        insertion_cost = option['insertion_cost']
        revenue = customer.revenue
        profit = revenue - insertion_cost
        
        return profit
    
    # ========================================================================
    # STEP 4: EXECUTE INSERTION
    # ========================================================================
    
    def _execute_truck_insertion(self, customer, option: Dict):
        """Execute truck route insertion"""
        position = option['position']
        self.truck_route.insert(position, customer.id)
    
    def _execute_drone_insertion(self, customer, option: Dict):
        """Execute drone trip insertion (create new trip)"""
        new_trip = {
            'customers': option['trip_customers'],
            'launch_node': 0,  # Simplified: launch from depot
            'retrieval_node': 0,
            'total_demand': customer.demand
        }
        self.drone_trips.append(new_trip)
    
    # ========================================================================
    # COST CALCULATION UTILITIES
    # ========================================================================
    
    def _calculate_truck_insertion_cost(self, customer, position: int) -> float:
        """
        Calculate time cost increase to insert customer at position
        
        Args:
            customer: Customer to insert
            position: Position in truck route (0 = first, n = last)
            
        Returns:
            Additional time cost
        """
        # Simplified cost: distance-based
        if position == 0:
            prev_loc = self.depot
        else:
            prev_cid = self.truck_route[position - 1]
            prev_loc = self.all_customers[prev_cid].location
        
        if position == len(self.truck_route):
            next_loc = self.depot
        else:
            next_cid = self.truck_route[position]
            next_loc = self.all_customers[next_cid].location
        
        # Current distance without insertion
        current_distance = self._calculate_distance(prev_loc, next_loc)
        
        # Distance with insertion
        new_distance = (
            self._calculate_distance(prev_loc, customer.location) +
            self._calculate_distance(customer.location, next_loc)
        )
        
        # Additional distance and cost
        added_distance = new_distance - current_distance
        added_time = added_distance / self.truck_speed
        added_cost = added_time * self.truck_cost_per_time
        
        return added_cost
    
    # NOTE: modified to accept optional launch/retrieval nodes. The original
    # implementation accepted only `customer_ids` (assumed launch/retrieve at
    # the depot). During refactor we need to compute drone operation cost when
    # launching from or retrieving to a truck node (not always the depot).
    #
    # This helper now forwards the provided `launch_node` and `retrieval_node`
    # to `_calculate_drone_insertion_cost_with_nodes(...)` which performs the
    # actual distance/time based cost computation. Keeping this thin wrapper
    # preserves backward compatibility for calls that only provide
    # `customer_ids`.
    def _calculate_drone_insertion_cost(self, customer_ids: List[int], launch_node: int = 0,
                                        retrieval_node: int = 0) -> float:
        """
        Calculate time cost to create drone trip.

        Supports optional `launch_node` and `retrieval_node` to compute
        trip cost when launching/retrieving at specific truck nodes.
        """
        return self._calculate_drone_insertion_cost_with_nodes(customer_ids, launch_node, retrieval_node)

    def _calculate_drone_insertion_cost_with_nodes(self, customer_ids: List[int],
                                                   launch_node: int, retrieval_node: int) -> float:
        """
        Calculate time cost to create drone trip when launching from `launch_node` and retrieving at `retrieval_node`.
        """
        trip_distance = 0.0
        current_loc = self.depot if launch_node == 0 else self.all_customers[launch_node].location

        for cid in customer_ids:
            customer = self.all_customers[cid]
            trip_distance += self._calculate_distance(current_loc, customer.location)
            current_loc = customer.location

        # to retrieval node
        retrieval_loc = self.depot if retrieval_node == 0 else self.all_customers[retrieval_node].location
        trip_distance += self._calculate_distance(current_loc, retrieval_loc)

        # Calculate cost
        trip_time = trip_distance / self.drone_speed
        trip_cost = trip_time * self.drone_cost_per_time

        return trip_cost
    
    def _is_trip_battery_feasible(self, customer_ids: List[int]) -> bool:
        """Check if drone trip is battery-feasible"""
        if not customer_ids:
            return True
        
        total_distance = 0.0
        current_loc = self.depot
        
        for cid in customer_ids:
            customer = self.all_customers[cid]
            total_distance += self._calculate_distance(current_loc, customer.location)
            current_loc = customer.location
        
        total_distance += self._calculate_distance(current_loc, self.depot)
        
        energy_needed = total_distance * self.drone_speed
        
        return energy_needed <= self.drone_battery

    def _compute_truck_arrival_times(self, route: List[int], start_time: float = 0.0) -> Dict[int, float]:
        """
        Compute arrival times at each node (including depot as node 0) for the given truck route.
        Returns a mapping node_id -> arrival_time.
        """
        times: Dict[int, float] = {}
        current_time = start_time
        current_loc = self.depot
        times[0] = current_time

        for cid in route:
            customer = self.all_customers[cid]
            travel_time = self._calculate_distance(current_loc, customer.location) / self.truck_speed
            arrival_time = current_time + travel_time
            times[cid] = arrival_time
            buffer = getattr(self.mdp, 'buffer_time', 0.0)
            current_time = arrival_time + getattr(customer, 'service_time', 0.0) + buffer
            current_loc = customer.location

        # return-to-depot sentinel
        travel_back = self._calculate_distance(current_loc, self.depot) / self.truck_speed
        buffer = getattr(self.mdp, 'buffer_time', 0.0)
        times[0 + 0.1] = current_time + travel_back + buffer
        return times

    def _node_location(self, node_id: int):
        if node_id == 0:
            return self.depot
        return self.all_customers[node_id].location

    def _drone_trip_duration(self, launch_node: int, customer_ids: List[int], retrieval_node: int) -> float:
        total_distance = 0.0
        current_loc = self._node_location(launch_node)
        for cid in customer_ids:
            cust_loc = self.all_customers[cid].location
            total_distance += self._calculate_distance(current_loc, cust_loc)
            current_loc = cust_loc
        total_distance += self._calculate_distance(current_loc, self._node_location(retrieval_node))
        flight_time = total_distance / self.drone_speed
        service_time = sum(getattr(self.all_customers[cid], 'service_time', 0.0) for cid in customer_ids)
        return flight_time + service_time
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def _calculate_distance(self, loc1: Tuple[float, float], 
                           loc2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance"""
        return np.sqrt((loc1[0] - loc2[0])**2 + (loc1[1] - loc2[1])**2)
    
    def get_statistics(self) -> Dict:
        """
        Get request handling statistics
        
        Returns:
            Dictionary with metrics
        """
        if self.requests_processed == 0:
            acceptance_rate = 0.0
        else:
            acceptance_rate = (self.requests_accepted / self.requests_processed) * 100
        
        return {
            'requests_processed': self.requests_processed,
            'requests_accepted': self.requests_accepted,
            'requests_rejected': self.requests_rejected,
            'acceptance_rate': acceptance_rate,
            'total_accepted_profit': self.total_accepted_profit,
            'accepted_request_ids': self.accepted_request_ids
        }
    
    def get_current_plan(self) -> Dict:
        """
        Get updated operational plan
        
        Returns:
            Dictionary with current truck route and drone trips
        """
        return {
            'truck_route': self.truck_route.copy(),
            'drone_trips': [trip.copy() for trip in self.drone_trips]
        }