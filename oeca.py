"""
OECA (Offline Enhanced Construction Algorithm)
Generates optimal initial routes for scheduled deliveries.

Based on: Gu, Liu & Poon (2023)
"""

import numpy as np
from typing import List, Dict, Tuple, Set


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
    
    # ========================================================================
    # STEP 1: TSP HEURISTIC FOR INITIAL ROUTE
    # ========================================================================
    
    def _nearest_neighbor_tsp(self) -> List[int]:
        """
        Nearest Neighbor heuristic for constructing initial truck route
        
        This is a greedy TSP heuristic:
        1. Start at depot
        2. Repeatedly visit the nearest unvisited customer
        3. Continue until all customers are visited
        
        Time Complexity: O(n²) where n = number of customers
        
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

    def _compute_truck_arrival_times(self, route: List[int], start_time: float = 0.0) -> Dict[int, float]:
        """
        Compute arrival times at each node (including depot as node 0) for the given truck route.

        Returns a mapping node_id -> arrival_time. Depot is represented by node id 0.
        """
        times: Dict[int, float] = {}
        current_time = start_time
        current_loc = self.depot
        times[0] = current_time

        for cid in route:
            customer = self.all_customers[cid]
            travel_time = self._calculate_distance(current_loc, customer.location) / self.truck_speed
            arrival_time = current_time + travel_time
            # account for service time at arrival and reserve buffer_time slack
            times[cid] = arrival_time
            buffer = getattr(self.mdp, 'buffer_time', 0.0)
            current_time = arrival_time + getattr(customer, 'service_time', 0.0) + buffer
            current_loc = customer.location

        # arrival at depot after finishing route
        travel_back = self._calculate_distance(current_loc, self.depot) / self.truck_speed
        buffer = getattr(self.mdp, 'buffer_time', 0.0)
        times[0 + 0.1] = current_time + travel_back + buffer  # sentinel for return-to-depot time
        return times

    def _node_location(self, node_id: int) -> Tuple[float, float]:
        """Return location tuple for a node id (0 => depot)."""
        if node_id == 0:
            return self.depot
        return self.all_customers[node_id].location

    def _drone_trip_duration(self, launch_node: int, customer_ids: List[int], retrieval_node: int) -> float:
        """
        Compute the drone trip duration (flight time + service times) when launched from `launch_node`,
        visiting `customer_ids` in order, and retrieving at `retrieval_node`.
        """
        total_distance = 0.0
        current_loc = self._node_location(launch_node)
        for cid in customer_ids:
            cust_loc = self.all_customers[cid].location
            total_distance += self._calculate_distance(current_loc, cust_loc)
            current_loc = cust_loc
        # to retrieval
        total_distance += self._calculate_distance(current_loc, self._node_location(retrieval_node))

        flight_time = total_distance / self.drone_speed
        service_time = sum(getattr(self.all_customers[cid], 'service_time', 0.0) for cid in customer_ids)
        return flight_time + service_time
    
    # ========================================================================
    # STEP 2: IDENTIFY DRONE CANDIDATES
    # ========================================================================
    
    def _identify_drone_candidates(self, route: List[int]) -> Set[int]:
        """
        Identify customers that would benefit from drone service
        
        A customer is a good drone candidate if:
        1. Demand ≤ drone capacity (physical constraint)
        2. Battery-feasible for drone to reach (energy constraint)
        3. Serving by drone saves time/cost (economic benefit)
        
        Args:
            route: Current truck route (list of customer IDs)
            
        Returns:
            Set of customer IDs suitable for drone service
        """
        candidates = set()
        
        print(f"    Evaluating {len(route)} customers for drone service...")
        
        # Precompute truck arrival times to evaluate synchronization
        truck_times = self._compute_truck_arrival_times(route)

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

            # CRITERION 3: Calculate if drone service provides benefit taking waiting into account
            # Determine launch/retrieval nodes (prev and next in route)
            idx = route.index(cid)
            if idx == 0:
                launch_node = 0
            else:
                launch_node = route[idx - 1]

            if idx == len(route) - 1:
                retrieval_node = 0
            else:
                retrieval_node = route[idx + 1]

            # Drone trip duration
            drone_duration = self._drone_trip_duration(launch_node, [cid], retrieval_node)

            # Truck time gap between launch and retrieval
            t_launch = truck_times.get(launch_node, truck_times.get(0, 0.0))
            t_retrieval = truck_times.get(retrieval_node, truck_times.get(0 + 0.1, 0.0))
            truck_gap = max(0.0, t_retrieval - t_launch)
            waiting_time = max(0.0, drone_duration - truck_gap)

            # detour savings (time truck would spend visiting customer)
            detour_savings = self._calculate_detour_savings(route, cid)

            # Convert waiting time to cost-equivalent time (truck waiting reduces savings)
            effective_savings = detour_savings - waiting_time

            # Only use drone if it provides meaningful net savings
            if effective_savings > 1.0:  # Threshold for minimum net savings
                candidates.add(cid)
                print(f"      Customer {cid}: OK drone candidate (net savings: {effective_savings:.2f})")
            else:
                print(f"      Customer {cid}: X insufficient net savings ({effective_savings:.2f})")
        
        return candidates
    
    def _calculate_detour_savings(self, route: List[int], customer_id: int) -> float:
        """
        Calculate time/cost saved if customer is served by drone vs truck
        
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
    
    # ========================================================================
    # STEP 3: ASSIGN DRONE TRIPS
    # ========================================================================
    
    def _assign_drone_trips(self, route: List[int], 
                           candidates: Set[int]) -> Tuple[List[int], List[Dict]]:
        """
        Assign drone candidates to multi-visit trips
        
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
        # Precompute truck arrival times on the original route to evaluate synchronization while grouping
        truck_times = self._compute_truck_arrival_times(route)
        
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
                
                # Determine actual launch/retrieval nodes (IDs already computed)
                trip = {
                    'customers': trip_customers,
                    'launch_node': launch_node,
                    'retrieval_node': retrieval_node,
                    'total_demand': trip_demand
                }
                # Verify time feasibility for this multi-customer trip
                drone_duration = self._drone_trip_duration(launch_node, trip_customers, retrieval_node)
                t_launch = truck_times.get(launch_node, truck_times.get(0, 0.0))
                t_retrieval = truck_times.get(retrieval_node, truck_times.get(0 + 0.1, 0.0))
                truck_gap = max(0.0, t_retrieval - t_launch)
                if drone_duration > (self.mdp.working_hours * 2):
                    # unrealistic long trip; skip
                    print(f"      Skipping trip {trip_customers}: unrealistic duration {drone_duration:.2f}")
                    i += 1
                    continue

                # Accept trip (truck may need to wait; cost handled later)
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
    
    # ========================================================================
    # STEP 4: EVALUATE SOLUTION
    # ========================================================================
    
    def _evaluate_solution(self, truck_route: List[int], 
                          drone_trips: List[Dict]) -> Tuple[float, float]:
        """
        Calculate total cost and revenue of the solution
        
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
        # Compute truck schedule times on the final truck route (used to compute waiting costs)
        truck_times = self._compute_truck_arrival_times(truck_route)
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
            
            # Calculate truck waiting cost induced by this drone trip
            launch_node = trip.get('launch_node', 0)
            retrieval_node = trip.get('retrieval_node', 0)
            drone_duration = self._drone_trip_duration(launch_node, trip['customers'], retrieval_node)
            t_launch = truck_times.get(launch_node, truck_times.get(0, 0.0))
            t_retrieval = truck_times.get(retrieval_node, truck_times.get(0 + 0.1, 0.0))
            truck_gap = max(0.0, t_retrieval - t_launch)
            waiting_time = max(0.0, drone_duration - truck_gap)
            waiting_cost = waiting_time * self.truck_cost_per_time
            if waiting_time > 0:
                print(f"      Trip {i}: truck waits {waiting_time:.2f} time units (cost ${waiting_cost:.2f})")
            total_cost += waiting_cost
            
            print(f"      Trip {i}: distance={trip_distance:.2f}, time={trip_time:.2f}, "
                  f"cost=${trip_cost:.2f}")
        
        return total_cost, total_revenue
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def _calculate_distance(self, loc1: Tuple[float, float], 
                           loc2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two locations"""
        return np.sqrt((loc1[0] - loc2[0])**2 + (loc1[1] - loc2[1])**2)
