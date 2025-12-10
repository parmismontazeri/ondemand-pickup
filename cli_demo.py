"""
CLI Demo: Integrated Truck-Drone MDP with OECA and Dynamic Request Handling

This script demonstrates the complete workflow:
1. Initialize problem with scheduled customers
2. Generate offline plan using OECA algorithm
3. Simulate online operations with dynamic requests
4. Report final results and statistics
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict

from truck_drone_mdp import TruckDroneMDP, Customer, State, Action, CustomerType, DroneStatus
from oeca import OECAPlanner
from dynamic_handler import DynamicRequestHandler


def create_demo_scenario() -> TruckDroneMDP:
    """
    Create a demo problem instance
    
    Returns:
        Configured TruckDroneMDP instance
    """
    # Add scheduled customers (known in advance)
    scheduled_customers = [
        Customer(id=1, location=(10, 5), demand=8.0, revenue=20.0, 
                customer_type=CustomerType.SCHEDULED_DELIVERY, deadline=100, time_window=(0, 100)),
        Customer(id=2, location=(15, 10), demand=5.0, revenue=15.0, 
                customer_type=CustomerType.SCHEDULED_DELIVERY, deadline=100, time_window=(0, 100)),
        Customer(id=3, location=(20, 8), demand=10.0, revenue=25.0, 
                customer_type=CustomerType.SCHEDULED_DELIVERY, deadline=100, time_window=(0, 100)),
        Customer(id=4, location=(12, 20), demand=6.0, revenue=18.0, 
                customer_type=CustomerType.SCHEDULED_DELIVERY, deadline=100, time_window=(0, 100)),
        Customer(id=5, location=(8, 15), demand=7.0, revenue=22.0, 
                customer_type=CustomerType.SCHEDULED_DELIVERY, deadline=100, time_window=(0, 100)),
        Customer(id=6, location=(18, 18), demand=9.0, revenue=24.0, 
                customer_type=CustomerType.SCHEDULED_DELIVERY, deadline=100, time_window=(0, 100)),
    ]
    
    # Create MDP with scheduled customers
    mdp = TruckDroneMDP(
        depot=(0, 0),
        scheduled_customers=scheduled_customers,
        truck_capacity=50.0,
        truck_speed=10.0,
        drone_capacity=15.0,
        drone_speed=15.0,
        drone_battery=100.0,
        truck_cost_per_time=2.0,
        drone_cost_per_time=1.5
    )
    
    # Add potential dynamic customers (will arrive as requests)
    dynamic_customers = [
        Customer(id=101, location=(14, 12), demand=5.0, revenue=15.0, 
                customer_type=CustomerType.ONDEMAND_PICKUP, deadline=80, time_window=(10, 80)),
        Customer(id=102, location=(11, 18), demand=4.0, revenue=12.0, 
                customer_type=CustomerType.ONDEMAND_PICKUP, deadline=85, time_window=(15, 85)),
        Customer(id=103, location=(16, 6), demand=6.0, revenue=18.0, 
                customer_type=CustomerType.ONDEMAND_PICKUP, deadline=90, time_window=(20, 90)),
    ]
    
    for customer in dynamic_customers:
        mdp.all_customers[customer.id] = customer
    
    return mdp


def main():
    """Main demo workflow"""
    
    print("\n" + "="*80)
    print("TRUCK-DRONE MDP INTEGRATED DEMO")
    print("="*80)
    print("This demo shows:")
    print("1. Offline planning with OECA for scheduled customers")
    print("2. Online dynamic request handling during operations")
    print("3. Integrated decision-making and profit analysis")
    print("="*80)
    
    # PHASE 1: SETUP
    print("\n" + "-"*80)
    print("PHASE 1: PROBLEM SETUP")
    print("-"*80)
    
    mdp = create_demo_scenario()
    print(f"OK Created MDP with {len(mdp.scheduled_customers)} scheduled customers")
    print(f"   Truck: capacity={mdp.truck_capacity}, speed={mdp.truck_speed}, "
          f"cost={mdp.truck_cost_per_time}/unit-time")
    print(f"   Drone: capacity={mdp.drone_capacity}, speed={mdp.drone_speed}, "
          f"battery={mdp.drone_battery}, cost={mdp.drone_cost_per_time}/unit-time")
    
    # PHASE 2: OFFLINE PLANNING (OECA)
    print("\n" + "-"*80)
    print("PHASE 2: OFFLINE PLANNING WITH OECA")
    print("-"*80)
    
    planner = OECAPlanner(mdp)
    initial_plan = planner.generate_initial_plan()
    
    initial_truck_route = initial_plan['truck_route']
    initial_drone_trips = initial_plan['drone_trips']
    initial_profit = initial_plan['total_profit']
    
    print(f"\nInitial plan summary:")
    print(f"  Truck route: {initial_truck_route} ({len(initial_truck_route)} customers)")
    print(f"  Drone trips: {len(initial_drone_trips)} trips "
          f"({sum(len(t['customers']) for t in initial_drone_trips)} customers)")
    print(f"  Initial profit: ${initial_profit:.2f}")
    
    # PHASE 3: CREATE INITIAL STATE AND HANDLER
    print("\n" + "-"*80)
    print("PHASE 3: INITIALIZE OPERATIONAL STATE")
    print("-"*80)
    
    # Create initial state
    initial_state = State(
        current_time=0,
        truck_location=mdp.depot,
        drone_location=mdp.depot,
        truck_load=0.0,
        drone_load=0.0,
        drone_status=DroneStatus.ON_TRUCK,
        drone_battery=mdp.drone_battery,
        served_customers=set(),
        unserved_customers=set(c.id for c in mdp.scheduled_customers),
        current_route=initial_truck_route.copy()
    )
    
    # Create dynamic request handler
    handler = DynamicRequestHandler(
        mdp, initial_state,
        initial_truck_route, initial_drone_trips
    )
    
    print("OK Operational state initialized")
    print(f"   Current time: {initial_state.current_time}")
    print(f"   Truck location: {initial_state.truck_location}")
    print(f"   Drone status: {initial_state.drone_status.value}")
    print(f"   Drone battery: {initial_state.drone_battery:.1f}")
    
    # PHASE 4: SIMULATE DYNAMIC REQUESTS
    print("\n" + "-"*80)
    print("PHASE 4: HANDLE DYNAMIC REQUESTS")
    print("-"*80)
    
    # Get dynamic customers
    dynamic_requests = [
        mdp.all_customers[101],
        mdp.all_customers[102],
        mdp.all_customers[103],
    ]
    
    dynamic_results = []
    for request_customer in dynamic_requests:
        result = handler.handle_request(request_customer)
        dynamic_results.append(result)
    
    # PHASE 5: FINAL REPORT
    print("\n" + "-"*80)
    print("PHASE 5: FINAL RESULTS AND STATISTICS")
    print("-"*80)
    
    stats = handler.get_statistics()
    final_plan = handler.get_current_plan()
    
    print(f"\nDynamic request handling statistics:")
    print(f"  Requests processed: {stats['requests_processed']}")
    print(f"  Requests accepted: {stats['requests_accepted']}")
    print(f"  Requests rejected: {stats['requests_rejected']}")
    print(f"  Acceptance rate: {stats['acceptance_rate']:.1f}%")
    print(f"  Total dynamic profit: ${stats['total_accepted_profit']:.2f}")
    
    print(f"\nFinal operational plan:")
    print(f"  Truck route: {final_plan['truck_route']} "
          f"({len(final_plan['truck_route'])} customers)")
    print(f"  Drone trips: {len(final_plan['drone_trips'])} trips")
    for i, trip in enumerate(final_plan['drone_trips'], 1):
        print(f"    Trip {i}: {trip['customers']} (demand: {trip['total_demand']:.1f})")
    
    total_profit = initial_profit + stats['total_accepted_profit']
    
    print(f"\nProfit analysis:")
    print(f"  Initial plan profit: ${initial_profit:.2f}")
    print(f"  Dynamic requests profit: ${stats['total_accepted_profit']:.2f}")
    print(f"  TOTAL PROFIT: ${total_profit:.2f}")
    
    # Show individual request results
    print(f"\nDynamic request details:")
    for i, result in enumerate(dynamic_results, 1):
        status = "ACCEPTED" if result['accepted'] else "REJECTED"
        print(f"  Request {i} (ID {result['request_id']}): {status}", end="")
        if result['accepted']:
            print(f" via {result['method'].upper()} (profit: ${result['profit']:.2f})")
        else:
            print(f" ({result['reason']})")
    
    print("\n" + "="*80)
    print("DEMO COMPLETE")
    print("="*80 + "\n")
    
    return {
        'initial_plan': initial_plan,
        'dynamic_stats': stats,
        'final_plan': final_plan,
        'total_profit': total_profit
    }


if __name__ == "__main__":
    results = main()
