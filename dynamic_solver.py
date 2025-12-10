"""
OECA (Offline Enhanced Construction Algorithm) Integration with MDP
Based on: Gu, Liu & Poon (2023)

This module integrates OECA with the existing TruckDroneMDP framework:
1. OECA generates initial routes for scheduled deliveries (offline phase)
2. MDP framework handles execution and state transitions
3. Placeholder for future dynamic request handling

Usage:
    from truck_drone_mdp import TruckDroneMDP, Customer, CustomerType
    from oeca_integration import OECAPlanner
    
    # Create MDP instance
    mdp = TruckDroneMDP(...)
    
    # Generate initial solution with OECA
    planner = OECAPlanner(mdp)
    solution = planner.generate_initial_plan()
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from copy import deepcopy

# Assumes you have the base MDP code available
# from truck_drone_mdp import TruckDroneMDP, Customer, State, Action, CustomerType
from truck_drone_mdp import TruckDroneMDP, Customer, State, Action, CustomerType, DroneStatus


# ============================================================================
# OECA - OFFLINE ENHANCED CONSTRUCTION ALGORITHM
# ============================================================================

class OECAPlanner:
    """
    OECA: Offline Enhanced Construction Algorithm
    
    Generates optimal initial routes for scheduled deliveries by:
    1. Building truck-only route using TSP heuristic
    2. Identifying customers suitable for drone service
    3. Assigning profitable customers to drone trips
    4. Optimizing truck-drone synchronization
    
    This creates the baseline plan before operations begin.
    """
    
    def __init__(self, mdp):
        """
        Initialize OECA planner with MDP instance
        
        Args:
            mdp: TruckDroneMDP instance containing problem parameters
        """
        self.mdp = mdp
        self.depot = mdp.depot
        self.customers = mdp.scheduled_customers
        self.all_customers = mdp.all_customers
        
        # Vehicle parameters from MDP
        self.truck_capacity = mdp.truck_capacity
        self.drone_capacity = mdp.drone_capacity
        self.truck_speed = mdp.truck_speed
        self.drone_speed = mdp.drone_speed
        self.drone_battery = mdp.drone_battery
        
        # Cost parameters from MDP
        self.truck_cost_per_time = mdp.truck_cost_per_time
        self.drone_cost_per_time = mdp.drone_cost_per_time
        
    def generate_initial_plan(self) -> Dict:
        """
        Main OECA algorithm: Generate complete initial solution
        
        This is the offline planning phase that happens BEFORE operations start.
        It creates an optimized plan for serving all scheduled deliveries.
        
        Returns:
            Dictionary containing:
                - 'truck_route': List of customer IDs for truck (in order)
                - 'drone_trips': List of drone trip dictionaries
                - 'total_cost': Estimated total operating cost
                - 'total_revenue': Expected total revenue
                - 'total_profit': Net profit (revenue - cost)
                - 'statistics': Additional metrics
        """
        print("\n" + "="*70)
        print("OECA: GENERATING INITIAL PLAN FOR SCHEDULED DELIVERIES")
        print("="*70)
        print(f"Total customers: {len(self.customers)}")
        print(f"Depot location: {self.depot}")
        
        # STEP 1: Build initial truck-only route using nearest neighbor TSP
        print("\n[Step 1/4] Building initial truck-only route (TSP)...")
        initial_route = self._nearest_neighbor_tsp()
        print(f"  OK Initial route created: {initial_route}")
        print(f"  OK Route length: {len(initial_route)} customers")
        
        # STEP 2: Identify which customers are good candidates for drone service
        print("\n[Step 2/4] Identifying drone service candidates...")
        drone_candidates = self._identify_drone_candidates(initial_route)
        print(f"  OK Found {len(drone_candidates)} drone candidates")
        if drone_candidates:
            print(f"  OK Candidate IDs: {sorted(drone_candidates)}")
        
        # STEP 3: Assign candidates to drone trips and update truck route
        print("\n[Step 3/4] Assigning customers to drone trips...")
        truck_route, drone_trips = self._assign_drone_trips(initial_route, drone_candidates)
        print(f"  OK Final truck route ({len(truck_route)} customers): {truck_route}")
        print(f"  OK Created {len(drone_trips)} drone trips")
        
        for i, trip in enumerate(drone_trips, 1):
            print(f"      Trip {i}: customers {trip['customers']}")
            print(f"              launch from node {trip['launch_node']}, "
                  f"retrieve at node {trip['retrieval_node']}")
        
        # STEP 4: Evaluate solution quality
        print("\n[Step 4/4] Evaluating solution quality...")
        total_cost, total_revenue = self._evaluate_solution(truck_route, drone_trips)
        total_profit = total_revenue - total_cost
        
        print(f"  OK Total revenue: ${total_revenue:.2f}")
        print(f"  OK Total cost: ${total_cost:.2f}")
        print(f"  OK Net profit: ${total_profit:.2f}")
        
        # Package solution
        solution = {
            'truck_route': truck_route,
            'drone_trips': drone_trips,
            'total_cost': total_cost,
            'total_revenue': total_revenue,
            'total_profit': total_profit,
            'statistics': {
                'initial_route_length': len(initial_route),
                'final_truck_customers': len(truck_route),
                'drone_trips_count': len(drone_trips),
                'drone_customers': sum(len(trip['customers']) for trip in drone_trips),
                'candidates_identified': len(drone_candidates)
            }
        }
        
        print("\n" + "="*70)
        print("INITIAL PLAN GENERATION COMPLETE")
        print("="*70)
        
        return solution
    
    # ------------------------------------------------------------------------
    # STEP 1: TSP HEURISTIC FOR INITIAL ROUTE
    # ------------------------------------------------------------------------
    
    def _nearest_neighbor_tsp(self) -> List[int]:
        """
        Nearest Neighbor heuristic for constructing initial truck route
        
        This is a greedy TSP heuristic:
        1. Start at depot
        2. Repeatedly visit the nearest unvisited customer
        3. Continue until all customers are visited
        
        Time Complexity: O(n²) where n = number of customers
        
        Not optimal, but:
        - Fast and simple
        - Produces reasonable solutions (typically within 25% of optimal)
        - Good starting point for further optimization
        
        Returns:
            List of customer IDs in visit order
        """
        route = []
        unvisited = set(c.id for c in self.customers)
        current_location = self.depot
        
        print(f"    Starting nearest neighbor from depot {self.depot}")
        
        while unvisited:
            # Find nearest unvisited customer to current location
            nearest_id = None
            min_distance = float('inf')
            
            for cid in unvisited:
                customer = self.all_customers[cid]
                dist = self._calculate_distance(current_location, customer.location)
                
                if dist < min_distance:
                    min_distance = dist
                    nearest_id = cid
            
            # Add nearest customer to route
            route.append(nearest_id)
            unvisited.remove(nearest_id)
            
            # Update current location
            current_location = self.all_customers[nearest_id].location
            
            print(f"    Added customer {nearest_id} (distance: {min_distance:.2f})")
        
        return route
    
    # ------------------------------------------------------------------------
    # STEP 2: IDENTIFY DRONE CANDIDATES
    # ------------------------------------------------------------------------
    
    def _identify_drone_candidates(self, route: List[int]) -> Set[int]:
        """
        Identify customers that would benefit from drone service
        
        A customer is a good drone candidate if:
        1. Demand ≤ drone capacity (physical constraint)
        2. Battery-feasible for drone to reach (energy constraint)
        3. Serving by drone saves time/cost (economic benefit)
        
        The key insight: If customer creates a large "detour" in truck route,
        drone can serve it while truck takes a more direct path.
        
        Args:
            route: Current truck route (list of customer IDs)
            
        Returns:
            Set of customer IDs suitable for drone service
        """
        candidates = set()
        
        print(f"    Evaluating {len(route)} customers for drone service...")
        
        for cid in route:
            customer = self.all_customers[cid]
            
            # CRITERION 1: Must fit in drone capacity
            if customer.demand > self.drone_capacity:
                print(f"      Customer {cid}: X demand {customer.demand} > "
                    f"drone capacity {self.drone_capacity}")
                continue
            
            # CRITERION 2: Must be battery-feasible
            if not self._is_battery_feasible_single(customer):
                print(f"      Customer {cid}: X not battery feasible")
                continue
            
            # CRITERION 3: Calculate if drone service provides benefit
            detour_savings = self._calculate_detour_savings(route, cid)
            
            # Only use drone if it provides meaningful savings
            if detour_savings > 1.0:  # Threshold for minimum savings
                candidates.add(cid)
                print(f"      Customer {cid}: OK drone candidate "
                    f"(savings: {detour_savings:.2f})")
            else:
                print(f"      Customer {cid}: X insufficient savings "
                    f"({detour_savings:.2f})")
        
        return candidates
    
    def _calculate_detour_savings(self, route: List[int], customer_id: int) -> float:
        """
        Calculate time/cost saved if customer is served by drone vs truck
        
        Savings calculation:
        - Truck path WITH customer: prev → customer → next
        - Truck path WITHOUT customer: prev → next (direct)
        - Savings = (with_customer - without_customer) / truck_speed
        
        Positive savings means drone service is beneficial.
        
        Args:
            route: Current truck route
            customer_id: Customer to evaluate
            
        Returns:
            Time savings (positive = beneficial to use drone)
        """
        try:
            idx = route.index(customer_id)
        except ValueError:
            return 0.0
        
        # Get locations of previous and next customers in route
        if idx == 0:
            prev_loc = self.depot
        else:
            prev_loc = self.all_customers[route[idx-1]].location
        
        if idx == len(route) - 1:
            next_loc = self.depot
        else:
            next_loc = self.all_customers[route[idx+1]].location
        
        customer_loc = self.all_customers[customer_id].location
        
        # Distance if truck visits this customer (detour)
        dist_with_customer = (
            self._calculate_distance(prev_loc, customer_loc) +
            self._calculate_distance(customer_loc, next_loc)
        )
        
        # Distance if truck skips this customer (direct)
        dist_without_customer = self._calculate_distance(prev_loc, next_loc)
        
        # Calculate time savings
        time_with = dist_with_customer / self.truck_speed
        time_without = dist_without_customer / self.truck_speed
        savings = time_with - time_without
        
        return savings
    
    def _is_battery_feasible_single(self, customer) -> bool:
        """
        Check if drone can serve a single customer (round trip from depot)
        
        Simplified check: depot → customer → depot
        In practice, launch/retrieval points may differ from depot.
        
        Args:
            customer: Customer to check
            
        Returns:
            True if battery-feasible
        """
        # Round trip distance
        round_trip_distance = 2 * self._calculate_distance(self.depot, customer.location)
        
        # Energy needed (simplified: distance * speed factor)
        energy_needed = round_trip_distance * self.drone_speed
        
        # Check against battery capacity
        return energy_needed <= self.drone_battery
    
    # ------------------------------------------------------------------------
    # STEP 3: ASSIGN DRONE TRIPS
    # ------------------------------------------------------------------------
    
    def _assign_drone_trips(self, route: List[int], 
                           candidates: Set[int]) -> Tuple[List[int], List[Dict]]:
        """
        Assign drone candidates to multi-visit trips
        
        Strategy:
        1. Group nearby candidates into trips (up to 3 customers per trip)
        2. Ensure each trip satisfies:
           - Total demand ≤ drone capacity
           - Total distance ≤ battery capacity
        3. Determine launch and retrieval points for synchronization
        4. Remove assigned customers from truck route
        
        The key challenge: Balance between:
        - Longer trips (serve more customers per flight)
        - Feasibility constraints (capacity, battery)
        - Synchronization efficiency (truck doesn't wait too long)
        
        Args:
            route: Initial truck route
            candidates: Customers suitable for drone service
            
        Returns:
            Tuple of (updated_truck_route, list_of_drone_trips)
        """
        drone_trips = []
        assigned_to_drone = set()
        
        # Get candidates in route order (maintains spatial coherence)
        sorted_candidates = [cid for cid in route if cid in candidates]
        
        print(f"    Grouping {len(sorted_candidates)} candidates into trips...")
        
        # Greedy grouping: try to combine consecutive candidates
        i = 0
        while i < len(sorted_candidates):
            trip_customers = []
            trip_demand = 0.0
            
            # Try to add up to 3 consecutive candidates to same trip
            for j in range(i, min(i + 3, len(sorted_candidates))):
                cid = sorted_candidates[j]
                customer = self.all_customers[cid]
                
                # Check if adding this customer keeps trip feasible
                new_demand = trip_demand + customer.demand
                new_trip = trip_customers + [cid]
                
                if (new_demand <= self.drone_capacity and
                    self._is_trip_battery_feasible(new_trip)):
                    # Feasible - add to trip
                    trip_customers.append(cid)
                    trip_demand = new_demand
                else:
                    # Not feasible - stop growing this trip
                    break
            
            # Create drone trip if we found any customers
            if trip_customers:
                # Determine launch point (customer before first in trip)
                trip_start_idx = route.index(trip_customers[0])
                if trip_start_idx > 0:
                    launch_node = route[trip_start_idx - 1]
                else:
                    launch_node = 0  # Launch from depot
                
                # Determine retrieval point (customer after last in trip)
                trip_end_idx = route.index(trip_customers[-1])
                if trip_end_idx < len(route) - 1:
                    retrieval_node = route[trip_end_idx + 1]
                else:
                    retrieval_node = 0  # Retrieve at depot
                
                # Create trip
                trip = {
                    'customers': trip_customers,
                    'launch_node': launch_node,
                    'retrieval_node': retrieval_node,
                    'total_demand': trip_demand
                }
                
                drone_trips.append(trip)
                assigned_to_drone.update(trip_customers)
                
                print(f"      Created trip: {trip_customers} "
                      f"(demand: {trip_demand:.1f})")
                
                # Move to next unassigned candidate
                i += len(trip_customers)
            else:
                # Couldn't create trip, move to next candidate
                i += 1
        
        # Remove drone-assigned customers from truck route
        updated_truck_route = [cid for cid in route if cid not in assigned_to_drone]
        
        print(f"    OK Assigned {len(assigned_to_drone)} customers to {len(drone_trips)} trips")
        print(f"    OK Remaining truck route: {len(updated_truck_route)} customers")
        
        return updated_truck_route, drone_trips
    
    def _is_trip_battery_feasible(self, customer_ids: List[int]) -> bool:
        """
        Check if multi-customer drone trip is battery-feasible
        
        Calculates total trip distance:
        depot → customer1 → customer2 → ... → customerN → depot
        
        Args:
            customer_ids: List of customers in trip (in order)
            
        Returns:
            True if total energy needed ≤ battery capacity
        """
        if not customer_ids:
            return True
        
        # Calculate total trip distance
        total_distance = 0.0
        current_loc = self.depot
        
        for cid in customer_ids:
            customer = self.all_customers[cid]
            total_distance += self._calculate_distance(current_loc, customer.location)
            current_loc = customer.location
        
        # Return to depot
        total_distance += self._calculate_distance(current_loc, self.depot)
        
        # Check energy requirement
        energy_needed = total_distance * self.drone_speed
        
        return energy_needed <= self.drone_battery
    
    # ------------------------------------------------------------------------
    # STEP 4: EVALUATE SOLUTION
    # ------------------------------------------------------------------------
    
    def _evaluate_solution(self, truck_route: List[int], 
                          drone_trips: List[Dict]) -> Tuple[float, float]:
        """
        Calculate total cost and revenue of the solution
        
        Costs include:
        - Truck travel time × truck cost rate
        - Drone flight time × drone cost rate
        
        Revenue:
        - Sum of all customer revenues
        
        Args:
            truck_route: Final truck route (customer IDs)
            drone_trips: List of drone trip dictionaries
            
        Returns:
            Tuple of (total_cost, total_revenue)
        """
        total_cost = 0.0
        total_revenue = 0.0
        
        # TRUCK COSTS AND REVENUE
        print(f"    Calculating truck route cost...")
        current_loc = self.depot
        truck_distance = 0.0
        
        for cid in truck_route:
            customer = self.all_customers[cid]
            
            # Add travel distance
            distance = self._calculate_distance(current_loc, customer.location)
            truck_distance += distance
            current_loc = customer.location
            
            # Add revenue
            total_revenue += customer.revenue
        
        # Return to depot
        truck_distance += self._calculate_distance(current_loc, self.depot)
        
        # Calculate truck cost (time × rate)
        truck_time = truck_distance / self.truck_speed
        truck_cost = truck_time * self.truck_cost_per_time
        total_cost += truck_cost
        
        print(f"      Truck: distance={truck_distance:.2f}, time={truck_time:.2f}, "
              f"cost=${truck_cost:.2f}")
        
        # DRONE COSTS AND REVENUE
        print(f"    Calculating drone trips cost...")
        for i, trip in enumerate(drone_trips, 1):
            trip_distance = 0.0
            current_loc = self.depot
            
            for cid in trip['customers']:
                customer = self.all_customers[cid]
                
                # Add travel distance
                distance = self._calculate_distance(current_loc, customer.location)
                trip_distance += distance
                current_loc = customer.location
                
                # Add revenue
                total_revenue += customer.revenue
            
            # Return to depot
            trip_distance += self._calculate_distance(current_loc, self.depot)
            
            # Calculate trip cost
            trip_time = trip_distance / self.drone_speed
            trip_cost = trip_time * self.drone_cost_per_time
            total_cost += trip_cost
            
            print(f"      Trip {i}: distance={trip_distance:.2f}, time={trip_time:.2f}, "
                  f"cost=${trip_cost:.2f}")
        
        return total_cost, total_revenue
    
    # ------------------------------------------------------------------------
    # UTILITY METHODS
    # ------------------------------------------------------------------------
    
    def _calculate_distance(self, loc1: Tuple[float, float], 
                           loc2: Tuple[float, float]) -> float:
        """
        Calculate Euclidean distance between two locations
        
        Args:
            loc1: First location (x, y)
            loc2: Second location (x, y)
            
        Returns:
            Distance between points
        """
        return np.sqrt((loc1[0] - loc2[0])**2 + (loc1[1] - loc2[1])**2)


# ============================================================================
# DYNAMIC REQUEST HANDLER (ONLINE PHASE)
# ============================================================================

class DynamicRequestHandler:
    """
    Handler for on-demand pickup requests that arrive during execution
    
    This implements the ONLINE PHASE of the D-TDRP-SDOP problem:
    - Receives on-demand pickup requests as they arrive in real-time
    - Evaluates feasibility and profitability of accepting each request
    - Makes accept/reject decisions to maximize total profit
    - Re-optimizes routes dynamically when requests are accepted
    
    Decision Strategy:
    1. Check hard constraints (capacity, time, battery)
    2. Estimate profit impact using MDP lookahead
    3. Accept if expected profit increase > 0
    4. Insert request into current plan optimally
    
    Integration with MDP:
    - Uses mdp.get_feasible_actions() for valid insertions
    - Uses mdp.transition() to simulate state changes
    - Evaluates multiple insertion positions
    """
    
    def __init__(self, mdp, initial_plan):
        """
        Initialize dynamic request handler
        
        Args:
            mdp: TruckDroneMDP instance
            initial_plan: Initial plan from OECA (truck_route, drone_trips)
        """
        self.mdp = mdp
        self.initial_plan = initial_plan
        
        # Track request decisions
        self.accepted_requests = []
        self.rejected_requests = []
        
        # Current execution state
        self.current_truck_route = initial_plan['truck_route'].copy()
        self.current_drone_trips = initial_plan['drone_trips'].copy()
        
        # Performance tracking
        self.total_requests = 0
        self.acceptance_rate = 0.0
    
    def handle_new_request(self, request, current_state) -> Tuple[bool, Dict]:
        """
        Handle a new on-demand pickup request
        
        Decision Process:
        1. Validate request meets basic criteria
        2. Check feasibility constraints
        3. Find best insertion position (truck or drone)
        4. Estimate profit impact
        5. Accept if profitable, reject otherwise
        6. Update plan if accepted
        
        Args:
            request: Customer object for on-demand pickup
            current_state: Current MDP state (from execution)
            
        Returns:
            Tuple of (accept_decision, updated_plan)
                - accept_decision: True if accepted, False if rejected
                - updated_plan: New plan if accepted, current plan if rejected
        """
        self.total_requests += 1
        
        print(f"\n{'='*70}")
        print(f"DYNAMIC REQUEST #{self.total_requests}")
        print(f"{'='*70}")
        print(f"Customer ID: {request.id}")
        print(f"Location: {request.location}")
        print(f"Demand: {request.demand}")
        print(f"Revenue: ${request.revenue:.2f}")
        print(f"Deadline: {request.deadline if request.deadline else 'None'}")
        print(f"Current time: {current_state.current_time:.2f}")
        
        # STEP 1: Quick feasibility check
        print(f"\n[Step 1/5] Checking basic feasibility...")
        if not self._is_request_feasible(request, current_state):
            print(f"  X Request REJECTED - Failed feasibility check")
            self._reject_request(request, "infeasible")
            return False, self._get_current_plan()
        print(f"  OK Request is feasible")
        
        # STEP 2: Find best insertion options
        print(f"\n[Step 2/5] Finding insertion options...")
        insertion_options = self._find_insertion_options(request, current_state)
        
        if not insertion_options:
            print(f"  X Request REJECTED - No valid insertion positions")
            self._reject_request(request, "no_insertion")
            return False, self._get_current_plan()
        
        print(f"  OK Found {len(insertion_options)} insertion options")
        
        # STEP 3: Evaluate each insertion option
        print(f"\n[Step 3/5] Evaluating insertion options...")
        best_option = self._evaluate_insertion_options(
            insertion_options, request, current_state
        )
        
        if best_option is None:
            print(f"  X Request REJECTED - No profitable insertion")
            self._reject_request(request, "unprofitable")
            return False, self._get_current_plan()
        
        print(f"  OK Best option: {best_option['type']} insertion")
        print(f"    Expected profit increase: ${best_option['profit_delta']:.2f}")
        
        # STEP 4: Make accept/reject decision
        print(f"\n[Step 4/5] Making decision...")
        if best_option['profit_delta'] > 0:
            print(f"  OK ACCEPTING request (profitable)")
            accept_decision = True
        else:
            print(f"  X REJECTING request (not profitable enough)")
            self._reject_request(request, "low_profit")
            return False, self._get_current_plan()
        
        # STEP 5: Update plan if accepted
        print(f"\n[Step 5/5] Updating plan...")
        updated_plan = self._insert_request(best_option, request, current_state)
        self._accept_request(request, best_option)
        
        print(f"  OK Plan updated successfully")
        print(f"\n{'='*70}")
        print(f"REQUEST ACCEPTED")
        print(f"Acceptance rate: {self._calculate_acceptance_rate():.1f}%")
        print(f"{'='*70}")
        
        return True, updated_plan
    
    # ------------------------------------------------------------------------
    # FEASIBILITY CHECKING
    # ------------------------------------------------------------------------
    
    def _is_request_feasible(self, request, current_state) -> bool:
        """
        Check if request meets basic feasibility constraints
        
        Hard Constraints:
        1. Time: Can reach customer before deadline
        2. Capacity: Demand fits in truck OR drone
        3. Working hours: Enough time remaining to serve
        
        Args:
            request: Customer request
            current_state: Current MDP state
            
        Returns:
            True if feasible
        """
        # CONSTRAINT 1: Deadline feasibility
        if request.deadline:
            # Estimate minimum time to reach customer
            min_travel_time = self._estimate_min_travel_time(
                current_state.truck_location, request.location
            )
            earliest_arrival = current_state.current_time + min_travel_time
            
            if earliest_arrival > request.deadline:
                print(f"    X Cannot meet deadline: arrival {earliest_arrival:.2f} > "
                    f"deadline {request.deadline:.2f}")
                return False
        
                # CONSTRAINT 2: Capacity feasibility
                if request.demand > self.mdp.truck_capacity and request.demand > self.mdp.drone_capacity:
                        print(f"    X Demand {request.demand} exceeds both truck "
                                    f"({self.mdp.truck_capacity}) and drone ({self.mdp.drone_capacity}) capacity")
                        return False
        
        # CONSTRAINT 3: Working hours
        if current_state.current_time >= self.mdp.working_hours * 0.9:
            print(f"    X Too close to end of working hours")
            return False
        
        return True
    
    def _estimate_min_travel_time(self, from_loc: Tuple[float, float], 
                                  to_loc: Tuple[float, float]) -> float:
        """
        Estimate minimum travel time to location (using faster vehicle)
        
        Args:
            from_loc: Starting location
            to_loc: Destination location
            
        Returns:
            Minimum travel time
        """
        distance = self._calculate_distance(from_loc, to_loc)
        # Use faster vehicle (drone) for lower bound estimate
        return distance / max(self.mdp.truck_speed, self.mdp.drone_speed)
    
    # ------------------------------------------------------------------------
    # INSERTION OPTION GENERATION
    # ------------------------------------------------------------------------
    
    def _find_insertion_options(self, request, current_state) -> List[Dict]:
        """
        Find all valid positions to insert request into current plan
        
        Options include:
        1. Insert into truck route (between any two customers or at end)
        2. Create new drone trip (if demand fits)
        3. Add to existing drone trip (if feasible)
        
        Args:
            request: Customer request
            current_state: Current MDP state
            
        Returns:
            List of insertion option dictionaries
        """
        options = []
        
        # OPTION TYPE 1: Insert into truck route
        truck_options = self._find_truck_insertions(request, current_state)
        options.extend(truck_options)
        print(f"    Found {len(truck_options)} truck insertion positions")
        
        # OPTION TYPE 2: Create new drone trip
        if request.demand <= self.mdp.drone_capacity:
            drone_option = self._create_drone_trip_option(request, current_state)
            if drone_option:
                options.append(drone_option)
                print(f"    Found 1 new drone trip option")
        
        # OPTION TYPE 3: Add to existing drone trip (simplified - skip for now)
        # This would require checking if request fits in existing trip
        
        return options
    
    def _find_truck_insertions(self, request, current_state) -> List[Dict]:
        """
        Find valid positions to insert request into truck route
        
        For each position in route:
        - Check if insertion maintains feasibility
        - Calculate detour cost
        - Estimate profit impact
        
        Args:
            request: Customer request
            current_state: Current MDP state
            
        Returns:
            List of truck insertion options
        """
        options = []
        
        # Get unserved customers in truck route
        remaining_route = [cid for cid in self.current_truck_route 
                          if cid in current_state.unserved_customers]
        
        # Try inserting at each position (including start and end)
        for insert_pos in range(len(remaining_route) + 1):
            # Check capacity
            if current_state.truck_load + request.demand > self.mdp.truck_capacity:
                continue
            
            # Create hypothetical route with insertion
            new_route = remaining_route[:insert_pos] + [request.id] + remaining_route[insert_pos:]
            
            # Calculate insertion cost
            insertion_cost = self._calculate_insertion_cost(
                request, insert_pos, remaining_route, current_state, 'truck'
            )
            
            # Create option
            option = {
                'type': 'truck',
                'position': insert_pos,
                'new_route': new_route,
                'insertion_cost': insertion_cost,
                'profit_delta': None  # Will be calculated later
            }
            
            options.append(option)
        
        return options
    
    def _create_drone_trip_option(self, request, current_state) -> Optional[Dict]:
        """
        Create option for serving request with new drone trip
        
        Args:
            request: Customer request
            current_state: Current MDP state
            
        Returns:
            Drone trip option dictionary or None if infeasible
        """
        # Check battery feasibility for round trip
        distance_to_customer = self._calculate_distance(
            current_state.truck_location, request.location
        )
        round_trip_distance = 2 * distance_to_customer  # Simplified
        energy_needed = round_trip_distance * self.mdp.drone_speed
        
        if energy_needed > current_state.drone_battery:
            return None
        
        # Check if drone is available
        if hasattr(current_state, 'drone_status'):
            if current_state.drone_status != DroneStatus.ON_TRUCK:
                return None
        
        # Calculate trip cost
        trip_time = round_trip_distance / self.mdp.drone_speed
        trip_cost = trip_time * self.mdp.drone_cost_per_time
        
        option = {
            'type': 'drone_trip',
            'customers': [request.id],
            'trip_cost': trip_cost,
            'trip_time': trip_time,
            'profit_delta': None  # Will be calculated later
        }
        
        return option
    
    def _calculate_insertion_cost(self, request, position, route, 
                                  current_state, vehicle_type) -> float:
        """
        Calculate additional cost of inserting request at position
        
        Cost = Additional travel distance × cost rate
        
        Args:
            request: Customer request
            position: Insertion position in route
            route: Current route
            current_state: Current state
            vehicle_type: 'truck' or 'drone'
            
        Returns:
            Additional cost
        """
        # Get locations before and after insertion point
        if position == 0:
            prev_loc = current_state.truck_location
        else:
            prev_customer = self.mdp.all_customers[route[position - 1]]
            prev_loc = prev_customer.location
        
        if position >= len(route):
            next_loc = self.mdp.depot
        else:
            next_customer = self.mdp.all_customers[route[position]]
            next_loc = next_customer.location
        
        # Calculate detour distance
        direct_distance = self._calculate_distance(prev_loc, next_loc)
        detour_distance = (self._calculate_distance(prev_loc, request.location) +
                          self._calculate_distance(request.location, next_loc))
        
        additional_distance = detour_distance - direct_distance
        
        # Calculate cost
        if vehicle_type == 'truck':
            speed = self.mdp.truck_speed
            cost_rate = self.mdp.truck_cost_per_time
        else:
            speed = self.mdp.drone_speed
            cost_rate = self.mdp.drone_cost_per_time
        
        additional_time = additional_distance / speed
        additional_cost = additional_time * cost_rate
        
        return additional_cost
    
    # ------------------------------------------------------------------------
    # OPTION EVALUATION
    # ------------------------------------------------------------------------
    
    def _evaluate_insertion_options(self, options: List[Dict], request, 
                                    current_state) -> Optional[Dict]:
        """
        Evaluate all insertion options and return the best one
        
        For each option:
        1. Calculate profit delta = revenue - insertion_cost
        2. Account for any penalties (late service, etc.)
        3. Choose option with highest profit delta
        
        Args:
            options: List of insertion options
            request: Customer request
            current_state: Current state
            
        Returns:
            Best option or None if none are profitable
        """
        best_option = None
        best_profit = float('-inf')
        
        for option in options:
            # Calculate profit delta
            revenue = request.revenue
            
            if option['type'] == 'truck':
                cost = option['insertion_cost']
            else:  # drone_trip
                cost = option['trip_cost']
            
            profit_delta = revenue - cost
            
            # Apply penalty if service would be late
            if request.deadline:
                # Estimate service time (simplified)
                estimated_service_time = current_state.current_time + 10  # Rough estimate
                if estimated_service_time > request.deadline:
                    profit_delta -= self.mdp.penalty_cost
            
            option['profit_delta'] = profit_delta
            
            # Track best option
            if profit_delta > best_profit:
                best_profit = profit_delta
                best_option = option
            
            print(f"      Option: {option['type']} - "
                  f"Revenue: ${revenue:.2f}, Cost: ${cost:.2f}, "
                  f"Profit: ${profit_delta:.2f}")
        
        # Only return if profitable
        if best_option and best_option['profit_delta'] > 0:
            return best_option
        
        return None
    
    # ------------------------------------------------------------------------
    # PLAN UPDATING
    # ------------------------------------------------------------------------
    
    def _insert_request(self, option: Dict, request, current_state) -> Dict:
        """
        Insert accepted request into current plan
        
        Updates:
        - Truck route or drone trips
        - All customers dictionary
        - Unserved customers set
        
        Args:
            option: Best insertion option
            request: Customer request
            current_state: Current state
            
        Returns:
            Updated plan dictionary
        """
        # Add request to MDP's customer database
        self.mdp.all_customers[request.id] = request
        
        if option['type'] == 'truck':
            # Insert into truck route
            self.current_truck_route = option['new_route']
            print(f"    Updated truck route: {self.current_truck_route}")
            
        elif option['type'] == 'drone_trip':
            # Add new drone trip
            new_trip = {
                'customers': option['customers'],
                'launch_node': current_state.truck_location,
                'retrieval_node': current_state.truck_location,
                'total_demand': request.demand
            }
            self.current_drone_trips.append(new_trip)
            print(f"    Added drone trip: {option['customers']}")
        
        return self._get_current_plan()
    
    def _get_current_plan(self) -> Dict:
        """Get current plan as dictionary"""
        return {
            'truck_route': self.current_truck_route,
            'drone_trips': self.current_drone_trips
        }
    
    # ------------------------------------------------------------------------
    # REQUEST TRACKING
    # ------------------------------------------------------------------------
    
    def _accept_request(self, request, option):
        """Record accepted request"""
        self.accepted_requests.append({
            'request': request,
            'option': option,
            'profit_delta': option['profit_delta']
        })
    
    def _reject_request(self, request, reason):
        """Record rejected request"""
        self.rejected_requests.append({
            'request': request,
            'reason': reason
        })
    
    def _calculate_acceptance_rate(self) -> float:
        """Calculate current acceptance rate"""
        if self.total_requests == 0:
            return 0.0
        return (len(self.accepted_requests) / self.total_requests) * 100
    
    def get_statistics(self) -> Dict:
        """
        Get statistics about request handling
        
        Returns:
            Dictionary with statistics
        """
        total_profit_from_requests = sum(
            req['profit_delta'] for req in self.accepted_requests
        )
        
        return {
            'total_requests': self.total_requests,
            'accepted': len(self.accepted_requests),
            'rejected': len(self.rejected_requests),
            'acceptance_rate': self._calculate_acceptance_rate(),
            'total_profit_from_requests': total_profit_from_requests,
            'rejection_reasons': self._get_rejection_breakdown()
        }
    
    def _get_rejection_breakdown(self) -> Dict:
        """Get breakdown of rejection reasons"""
        reasons = {}
        for rejection in self.rejected_requests:
            reason = rejection['reason']
            reasons[reason] = reasons.get(reason, 0) + 1
        return reasons
    
    # ------------------------------------------------------------------------
    # UTILITY METHODS
    # ------------------------------------------------------------------------
    
    def _calculate_distance(self, loc1: Tuple[float, float], 
                           loc2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two locations"""
        return np.sqrt((loc1[0] - loc2[0])**2 + (loc1[1] - loc2[1])**2)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """
    Complete demonstration: OECA + Dynamic Request Handling
    
    This shows the full workflow:
    1. Generate initial plan with OECA (offline phase)
    2. Handle dynamic on-demand requests (online phase)
    3. Track performance and statistics
    """
    
    # NOTE: This requires the truck_drone_mdp module
    # Uncomment the following line when integrating:
    # from truck_drone_mdp import TruckDroneMDP, Customer, CustomerType, State
    
    # For demonstration, using mock classes
    from dataclasses import dataclass
    from enum import Enum
    
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
        time_window: Tuple[float, float]
        revenue: float
        service_time: float = 2.0
        deadline: Optional[float] = None
    
    @dataclass
    class State:
        current_time: float
        truck_location: Tuple[float, float]
        drone_location: Tuple[float, float]
        drone_status: DroneStatus
        drone_battery: float
        truck_load: float
        drone_load: float
        unserved_customers: Set[int]
        served_customers: Set[int]
    
    # Mock MDP for demonstration
    class MockMDP:
        def __init__(self, depot, customers, truck_capacity, drone_capacity,
                     truck_speed, drone_speed, drone_battery,
                     truck_cost_per_time, drone_cost_per_time, penalty_cost,
                     working_hours):
            self.depot = depot
            self.scheduled_customers = customers
            self.all_customers = {c.id: c for c in customers}
            self.truck_capacity = truck_capacity
            self.drone_capacity = drone_capacity
            self.truck_speed = truck_speed
            self.drone_speed = drone_speed
            self.drone_battery = drone_battery
            self.truck_cost_per_time = truck_cost_per_time
            self.drone_cost_per_time = drone_cost_per_time
            self.penalty_cost = penalty_cost
            self.working_hours = working_hours
    
    print("\n" + "#"*70)
    print("# COMPLETE DEMONSTRATION: OECA + DYNAMIC REQUESTS")
    print("#"*70)
    
    # ========================================================================
    # PHASE 1: OFFLINE PLANNING (OECA)
    # ========================================================================
    
    print("\n" + "="*70)
    print("PHASE 1: OFFLINE PLANNING WITH OECA")
    print("="*70)
    
    # Create problem instance with SCHEDULED deliveries
    depot = (0.0, 0.0)
    scheduled_customers = [
        Customer(1, (10, 10), CustomerType.SCHEDULED_DELIVERY, 5, (0, 100), 50),
        Customer(2, (20, 5), CustomerType.SCHEDULED_DELIVERY, 3, (0, 100), 30),
        Customer(3, (15, 20), CustomerType.SCHEDULED_DELIVERY, 4, (0, 100), 40),
        Customer(4, (5, 25), CustomerType.SCHEDULED_DELIVERY, 2, (0, 100), 25),
        Customer(5, (30, 15), CustomerType.SCHEDULED_DELIVERY, 3, (0, 100), 35),
        Customer(6, (12, 30), CustomerType.SCHEDULED_DELIVERY, 2, (0, 100), 28),
    ]
    
    mdp = MockMDP(
        depot=depot,
        customers=scheduled_customers,
        truck_capacity=20,
        drone_capacity=8,
        truck_speed=1.0,
        drone_speed=2.0,
        drone_battery=100,
        truck_cost_per_time=1.0,
        drone_cost_per_time=0.5,
        penalty_cost=15.0,
        working_hours=100
    )
    
    # Generate initial plan with OECA
    planner = OECAPlanner(mdp)
    initial_solution = planner.generate_initial_plan()
    
    print("\n" + "="*70)
    print("INITIAL PLAN SUMMARY")
    print("="*70)
    print(f"Truck route: {initial_solution['truck_route']}")
    print(f"Drone trips: {len(initial_solution['drone_trips'])}")
    print(f"Initial profit: ${initial_solution['total_profit']:.2f}")
    
    # ========================================================================
    # PHASE 2: ONLINE EXECUTION WITH DYNAMIC REQUESTS
    # ========================================================================
    
    print("\n" + "="*70)
    print("PHASE 2: ONLINE EXECUTION - HANDLING DYNAMIC REQUESTS")
    print("="*70)
    
    # Initialize dynamic request handler
    handler = DynamicRequestHandler(mdp, initial_solution)
    
    # Create mock current state (simulating mid-execution)
    current_state = State(
        current_time=15.0,
        truck_location=(10, 10),
        drone_location=(10, 10),
        drone_status=DroneStatus.ON_TRUCK,
        drone_battery=100.0,
        truck_load=5.0,
        drone_load=0.0,
        unserved_customers={2, 3, 4, 5, 6},
        served_customers={1}
    )
    
    # Simulate on-demand pickup requests arriving
    ondemand_requests = [
        Customer(101, (18, 12), CustomerType.ONDEMAND_PICKUP, 
                3, (0, 100), 45, deadline=50),
        Customer(102, (22, 20), CustomerType.ONDEMAND_PICKUP, 
                2, (0, 100), 28, deadline=60),
        Customer(103, (8, 18), CustomerType.ONDEMAND_PICKUP, 
                6, (0, 100), 38, deadline=40),
    ]
    
    print(f"\nSimulating {len(ondemand_requests)} on-demand requests...\n")
    
    # Handle each request
    for request in ondemand_requests:
        accept, updated_plan = handler.handle_new_request(request, current_state)
        
        # Simulate time passing and state changes
        current_state.current_time += 5.0
        
        print()  # Blank line between requests
    
    # ========================================================================
    # FINAL RESULTS
    # ========================================================================
    
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    
    # Get statistics
    stats = handler.get_statistics()
    
    print(f"\nREQUEST HANDLING STATISTICS:")
    print(f"   Total requests received: {stats['total_requests']}")
    print(f"   Accepted: {stats['accepted']}")
    print(f"   Rejected: {stats['rejected']}")
    print(f"   Acceptance rate: {stats['acceptance_rate']:.1f}%")
    print(f"   Profit from dynamic requests: ${stats['total_profit_from_requests']:.2f}")
    
    print(f"\nREJECTION BREAKDOWN:")
    for reason, count in stats['rejection_reasons'].items():
        print(f"   {reason}: {count}")
    
    print(f"\nPROFIT SUMMARY:")
    print(f"   Initial plan profit: ${initial_solution['total_profit']:.2f}")
    print(f"   Dynamic requests profit: ${stats['total_profit_from_requests']:.2f}")
    total_profit = initial_solution['total_profit'] + stats['total_profit_from_requests']
    print(f"   TOTAL PROFIT: ${total_profit:.2f}")
    
    print(f"\nFINAL PLAN:")
    final_plan = handler._get_current_plan()
    print(f"   Truck route: {final_plan['truck_route']}")
    print(f"   Drone trips: {len(final_plan['drone_trips'])}")
    
    if stats['accepted'] > 0:
        print(f"\nACCEPTED REQUESTS:")
        for i, accepted in enumerate(handler.accepted_requests, 1):
            req = accepted['request']
            opt = accepted['option']
            print(f"   {i}. Customer {req.id} - {opt['type']} service "
                  f"(profit: ${accepted['profit_delta']:.2f})")
    
    if stats['rejected'] > 0:
        print(f"\nREJECTED REQUESTS:")
        for i, rejected in enumerate(handler.rejected_requests, 1):
            req = rejected['request']
            print(f"   {i}. Customer {req.id} - Reason: {rejected['reason']}")
    
    print("\n" + "#"*70)
    print("# DEMONSTRATION COMPLETE")
    print("#"*70)
    print("\nKey Features Demonstrated:")
    print("- OECA generates initial plan for scheduled deliveries")
    print("- Dynamic handler evaluates on-demand requests in real-time")
    print("- Accept/reject decisions based on profitability")
    print("- Route updates when requests are accepted")
    print("- Performance tracking and statistics")
    print("\nNext Steps:")
    print("1. Integrate with your TruckDroneMDP class")
    print("2. Add actual MDP state transitions during execution")
    print("3. Implement request arrival simulation (Poisson process)")
    print("4. Add more sophisticated insertion heuristics")