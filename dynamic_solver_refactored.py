"""
Dynamic Solver: Orchestrator for Truck-Drone MDP with OECA and Dynamic Requests

This module re-exports and demonstrates the modular components:
- OECAPlanner for offline planning (from oeca.py)
- DynamicRequestHandler for online request handling (from dynamic_handler.py)
- Integrated workflow for complete optimization

Usage:
    from dynamic_solver_refactored import OECAPlanner, DynamicRequestHandler
    from truck_drone_mdp import TruckDroneMDP, Customer, CustomerType
    
    # Create MDP
    mdp = TruckDroneMDP(...)
    
    # Offline planning
    planner = OECAPlanner(mdp)
    plan = planner.generate_initial_plan()
    
    # Online request handling
    handler = DynamicRequestHandler(mdp, initial_state, plan['truck_route'], plan['drone_trips'])
    result = handler.handle_request(new_customer)
"""

# Re-export modular components for backwards compatibility
from oeca import OECAPlanner
from dynamic_handler import DynamicRequestHandler

__all__ = ['OECAPlanner', 'DynamicRequestHandler']
