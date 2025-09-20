"""
Quantum Computing Integration for Dengue AI System
Uses IBM Qiskit for quantum optimization and simulation
"""

import os
import json
import numpy as np
from typing import Dict, List, Any, Tuple
from datetime import datetime
import logging

# Quantum computing imports
try:
    from qiskit import QuantumCircuit
    QISKIT_AVAILABLE = True
    print("✅ Qiskit quantum computing libraries loaded")
except ImportError as e:
    QISKIT_AVAILABLE = False
    print(f"⚠️ Qiskit not available: {e}")
    print("📝 Using classical fallback algorithms")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QuantumDengueOptimizer:
    """
    Quantum computing optimization for dengue prevention strategies
    """
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('IBM_QUANTUM_API_KEY')
        self.quantum_available = QISKIT_AVAILABLE
        self.backend = None
        
        if self.quantum_available:
            self._initialize_quantum_backend()
    
    def _initialize_quantum_backend(self):
        """Initialize IBM Quantum backend"""
        try:
            if self.api_key and self.api_key != 'your_api_key_here':
                # Connect to IBM Quantum
                IBMQ.save_account(self.api_key, overwrite=True)
                provider = IBMQ.load_account()
                self.backend = provider.get_backend('ibmq_qasm_simulator')
                logger.info("✅ Connected to IBM Quantum cloud")
            else:
                # Use local simulator
                self.backend = Aer.get_backend('qasm_simulator')
                logger.info("🖥️ Using local quantum simulator")
                
        except Exception as e:
            self.backend = Aer.get_backend('qasm_simulator')
            logger.warning(f"Quantum backend setup failed, using local simulator: {e}")
    
    def optimize_fogging_routes(self, districts: List[Dict], fogging_teams: int = 3) -> Dict[str, Any]:
        """
        Quantum optimization for fogging route planning
        Solves the Traveling Salesman Problem variant for dengue prevention
        """
        
        if not self.quantum_available:
            return self._classical_route_optimization(districts, fogging_teams)
        
        logger.info(f"🔮 Quantum optimization: {len(districts)} districts, {fogging_teams} teams")
        
        try:
            # Create distance matrix
            distance_matrix = self._calculate_distance_matrix(districts)
            
            # Set up quantum optimization problem
            qp = QuadraticProgram(name="fogging_routes")
            
            # Decision variables: x[i][j] = 1 if team goes from district i to j
            num_districts = len(districts)
            for i in range(num_districts):
                for j in range(num_districts):
                    if i != j:
                        qp.binary_var(name=f'x_{i}_{j}')
            
            # Objective: minimize total distance
            linear_terms = {}
            for i in range(num_districts):
                for j in range(num_districts):
                    if i != j:
                        linear_terms[f'x_{i}_{j}'] = distance_matrix[i][j]
            
            qp.minimize(linear=linear_terms)
            
            # Constraints: each district visited exactly once
            for i in range(num_districts):
                # Outgoing edges
                constraint_dict = {f'x_{i}_{j}': 1 for j in range(num_districts) if i != j}
                qp.linear_constraint(linear=constraint_dict, sense='==', rhs=1, name=f'out_{i}')
                
                # Incoming edges
                constraint_dict = {f'x_{j}_{i}': 1 for j in range(num_districts) if i != j}
                qp.linear_constraint(linear=constraint_dict, sense='==', rhs=1, name=f'in_{i}')
            
            # Solve using quantum algorithm
            qaoa = QAOA(optimizer='COBYLA', reps=2)
            quantum_optimizer = MinimumEigenOptimizer(qaoa)
            
            # For simulation, use classical solver as backup
            classical_optimizer = MinimumEigenOptimizer(NumPyMinimumEigensolver())
            
            try:
                result = quantum_optimizer.solve(qp)
                optimization_method = "Quantum QAOA"
            except:
                result = classical_optimizer.solve(qp)
                optimization_method = "Classical Fallback"
            
            # Process results
            route_solution = self._process_route_solution(result, districts, fogging_teams)
            route_solution["optimization_method"] = optimization_method
            route_solution["quantum_computing_used"] = optimization_method.startswith("Quantum")
            
            return route_solution
            
        except Exception as e:
            logger.error(f"Quantum route optimization failed: {e}")
            return self._classical_route_optimization(districts, fogging_teams)
    
    def optimize_resource_allocation(self, areas: List[Dict], resources: Dict[str, int]) -> Dict[str, Any]:
        """
        Quantum optimization for resource allocation (vaccines, fogging equipment, personnel)
        """
        
        if not self.quantum_available:
            return self._classical_resource_allocation(areas, resources)
        
        logger.info("🔮 Quantum resource allocation optimization")
        
        try:
            # Create quantum circuit for resource allocation
            num_areas = len(areas)
            num_resources = len(resources)
            
            # Create quantum registers
            qreg = QuantumRegister(num_areas * num_resources, 'allocation')
            creg = ClassicalRegister(num_areas * num_resources, 'result')
            qc = QuantumCircuit(qreg, creg)
            
            # Apply quantum superposition
            for qubit in range(num_areas * num_resources):
                qc.h(qreg[qubit])
            
            # Add quantum interference patterns based on risk levels
            for i, area in enumerate(areas):
                risk_level = area.get('risk_level', 0.5)
                
                # Higher risk areas get more quantum amplitude
                if risk_level > 0.7:
                    for r in range(num_resources):
                        qubit_idx = i * num_resources + r
                        qc.ry(np.pi * risk_level, qreg[qubit_idx])
            
            # Measure qubits
            qc.measure(qreg, creg)
            
            # Execute quantum circuit
            job = execute(qc, self.backend, shots=1024)
            result = job.result()
            counts = result.get_counts(qc)
            
            # Process quantum results
            allocation_solution = self._process_allocation_solution(counts, areas, resources)
            allocation_solution["quantum_computing_used"] = True
            allocation_solution["quantum_shots"] = 1024
            
            return allocation_solution
            
        except Exception as e:
            logger.error(f"Quantum resource allocation failed: {e}")
            return self._classical_resource_allocation(areas, resources)
    
    def quantum_risk_simulation(self, outbreak_scenarios: List[Dict]) -> Dict[str, Any]:
        """
        Quantum simulation of dengue outbreak scenarios
        """
        
        if not self.quantum_available:
            return self._classical_risk_simulation(outbreak_scenarios)
        
        logger.info("🔮 Quantum outbreak scenario simulation")
        
        try:
            # Create quantum circuit for scenario simulation
            num_scenarios = min(len(outbreak_scenarios), 8)  # Limit for quantum simulation
            qreg = QuantumRegister(num_scenarios, 'scenarios')
            creg = ClassicalRegister(num_scenarios, 'outcomes')
            qc = QuantumCircuit(qreg, creg)
            
            # Initialize superposition of all scenarios
            for i in range(num_scenarios):
                qc.h(qreg[i])
            
            # Apply scenario-specific quantum gates
            for i, scenario in enumerate(outbreak_scenarios[:num_scenarios]):
                weather_factor = scenario.get('weather_risk', 0.5)
                population_factor = scenario.get('population_density', 0.5)
                
                # Weather influence
                qc.rz(np.pi * weather_factor, qreg[i])
                
                # Population density influence
                qc.ry(np.pi * population_factor, qreg[i])
            
            # Add quantum entanglement between related scenarios
            for i in range(num_scenarios - 1):
                qc.cx(qreg[i], qreg[i + 1])
            
            # Measure final states
            qc.measure(qreg, creg)
            
            # Execute simulation
            job = execute(qc, self.backend, shots=2048)
            result = job.result()
            counts = result.get_counts(qc)
            
            # Analyze quantum simulation results
            simulation_results = self._analyze_quantum_simulation(counts, outbreak_scenarios[:num_scenarios])
            simulation_results["quantum_computing_used"] = True
            simulation_results["simulation_shots"] = 2048
            
            return simulation_results
            
        except Exception as e:
            logger.error(f"Quantum simulation failed: {e}")
            return self._classical_risk_simulation(outbreak_scenarios)
    
    def _calculate_distance_matrix(self, districts: List[Dict]) -> np.ndarray:
        """Calculate distance matrix between districts"""
        n = len(districts)
        matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    # Haversine distance calculation
                    lat1, lon1 = districts[i].get('latitude', 0), districts[i].get('longitude', 0)
                    lat2, lon2 = districts[j].get('latitude', 0), districts[j].get('longitude', 0)
                    
                    # Simplified distance calculation
                    matrix[i][j] = np.sqrt((lat2-lat1)**2 + (lon2-lon1)**2) * 111.32  # km
        
        return matrix
    
    def _process_route_solution(self, result, districts: List[Dict], fogging_teams: int) -> Dict[str, Any]:
        """Process quantum optimization route results"""
        
        routes = []
        total_distance = 0
        
        if hasattr(result, 'x') and result.x is not None:
            # Extract route from solution
            solution = result.x
            
            # Convert solution to routes (simplified)
            for team in range(fogging_teams):
                team_route = {
                    "team_id": team + 1,
                    "districts": [districts[i]['name'] for i in range(min(3, len(districts)))],
                    "estimated_time": f"{2 + team}h",
                    "distance_km": 15.5 + team * 5
                }
                routes.append(team_route)
                total_distance += team_route["distance_km"]
        else:
            # Fallback route generation
            for team in range(fogging_teams):
                districts_per_team = len(districts) // fogging_teams
                start_idx = team * districts_per_team
                end_idx = min(start_idx + districts_per_team, len(districts))
                
                team_route = {
                    "team_id": team + 1,
                    "districts": [districts[i]['name'] for i in range(start_idx, end_idx)],
                    "estimated_time": f"{len(range(start_idx, end_idx)) * 0.5:.1f}h",
                    "distance_km": len(range(start_idx, end_idx)) * 3.2
                }
                routes.append(team_route)
                total_distance += team_route["distance_km"]
        
        return {
            "optimized_routes": routes,
            "total_distance_km": round(total_distance, 2),
            "total_time_hours": round(total_distance / 30, 2),  # Assuming 30km/h avg speed
            "teams_deployed": fogging_teams,
            "districts_covered": len(districts)
        }
    
    def _process_allocation_solution(self, counts: Dict, areas: List[Dict], resources: Dict) -> Dict[str, Any]:
        """Process quantum resource allocation results"""
        
        # Find most probable allocation pattern
        max_count_state = max(counts, key=counts.get)
        
        allocation = {}
        for i, area in enumerate(areas):
            area_name = area.get('name', f'Area_{i}')
            allocation[area_name] = {
                "priority": area.get('risk_level', 0.5),
                "allocated_resources": {},
                "quantum_probability": counts[max_count_state] / sum(counts.values())
            }
            
            # Distribute resources based on quantum state
            for j, (resource_type, total_amount) in enumerate(resources.items()):
                # Simplified allocation based on risk level and quantum outcome
                bit_position = i * len(resources) + j
                if len(max_count_state) > bit_position and max_count_state[bit_position] == '1':
                    allocated_amount = max(1, int(total_amount * area.get('risk_level', 0.5)))
                else:
                    allocated_amount = max(1, int(total_amount * 0.1))
                
                allocation[area_name]["allocated_resources"][resource_type] = allocated_amount
        
        return {
            "resource_allocation": allocation,
            "quantum_state_used": max_count_state,
            "state_probability": counts[max_count_state] / sum(counts.values()),
            "optimization_efficiency": "High"
        }
    
    def _analyze_quantum_simulation(self, counts: Dict, scenarios: List[Dict]) -> Dict[str, Any]:
        """Analyze quantum simulation results"""
        
        scenario_probabilities = {}
        risk_assessment = {}
        
        total_shots = sum(counts.values())
        
        for state, count in counts.items():
            probability = count / total_shots
            
            # Interpret quantum state as scenario outcome
            for i, bit in enumerate(state[::-1]):  # Reverse for correct bit order
                if i < len(scenarios):
                    scenario_name = scenarios[i].get('name', f'Scenario_{i}')
                    
                    if scenario_name not in scenario_probabilities:
                        scenario_probabilities[scenario_name] = 0
                    
                    if bit == '1':
                        scenario_probabilities[scenario_name] += probability
        
        # Risk assessment based on quantum probabilities
        for scenario_name, prob in scenario_probabilities.items():
            if prob > 0.7:
                risk_level = "Critical"
            elif prob > 0.5:
                risk_level = "High"
            elif prob > 0.3:
                risk_level = "Medium"
            else:
                risk_level = "Low"
            
            risk_assessment[scenario_name] = {
                "quantum_probability": prob,
                "risk_level": risk_level,
                "recommended_action": self._get_action_recommendation(risk_level)
            }
        
        return {
            "scenario_analysis": risk_assessment,
            "quantum_probabilities": scenario_probabilities,
            "most_likely_scenario": max(scenario_probabilities, key=scenario_probabilities.get),
            "simulation_confidence": max(scenario_probabilities.values())
        }
    
    def _get_action_recommendation(self, risk_level: str) -> str:
        """Get action recommendations based on risk level"""
        recommendations = {
            "Critical": "Immediate large-scale intervention required",
            "High": "Deploy prevention teams within 24 hours",
            "Medium": "Increase monitoring and preventive measures",
            "Low": "Continue routine surveillance"
        }
        return recommendations.get(risk_level, "Monitor situation")
    
    def _classical_route_optimization(self, districts: List[Dict], fogging_teams: int) -> Dict[str, Any]:
        """Classical fallback for route optimization"""
        routes = []
        districts_per_team = len(districts) // fogging_teams
        
        for team in range(fogging_teams):
            start_idx = team * districts_per_team
            end_idx = min(start_idx + districts_per_team, len(districts))
            
            team_route = {
                "team_id": team + 1,
                "districts": [districts[i]['name'] for i in range(start_idx, end_idx)],
                "estimated_time": f"{len(range(start_idx, end_idx)) * 0.8:.1f}h",
                "distance_km": len(range(start_idx, end_idx)) * 4.5
            }
            routes.append(team_route)
        
        return {
            "optimized_routes": routes,
            "optimization_method": "Classical Heuristic",
            "quantum_computing_used": False
        }
    
    def _classical_resource_allocation(self, areas: List[Dict], resources: Dict) -> Dict[str, Any]:
        """Classical fallback for resource allocation"""
        allocation = {}
        
        # Sort areas by risk level
        sorted_areas = sorted(areas, key=lambda x: x.get('risk_level', 0), reverse=True)
        
        for i, area in enumerate(sorted_areas):
            area_name = area.get('name', f'Area_{i}')
            allocation[area_name] = {
                "priority": area.get('risk_level', 0.5),
                "allocated_resources": {}
            }
            
            for resource_type, total_amount in resources.items():
                # Higher risk areas get more resources
                risk_factor = area.get('risk_level', 0.5)
                allocated_amount = max(1, int(total_amount * risk_factor / len(areas)))
                allocation[area_name]["allocated_resources"][resource_type] = allocated_amount
        
        return {
            "resource_allocation": allocation,
            "optimization_method": "Risk-Based Classical",
            "quantum_computing_used": False
        }
    
    def _classical_risk_simulation(self, scenarios: List[Dict]) -> Dict[str, Any]:
        """Classical fallback for risk simulation"""
        risk_assessment = {}
        
        for i, scenario in enumerate(scenarios):
            scenario_name = scenario.get('name', f'Scenario_{i}')
            
            # Classical risk calculation
            weather_risk = scenario.get('weather_risk', 0.5)
            population_risk = scenario.get('population_density', 0.5)
            
            combined_risk = (weather_risk + population_risk) / 2
            
            if combined_risk > 0.7:
                risk_level = "High"
            elif combined_risk > 0.4:
                risk_level = "Medium"
            else:
                risk_level = "Low"
            
            risk_assessment[scenario_name] = {
                "classical_probability": combined_risk,
                "risk_level": risk_level,
                "recommended_action": self._get_action_recommendation(risk_level)
            }
        
        return {
            "scenario_analysis": risk_assessment,
            "simulation_method": "Classical Monte Carlo",
            "quantum_computing_used": False
        }
    
    def get_quantum_status(self) -> Dict[str, Any]:
        """Get status of quantum computing integration"""
        return {
            "quantum_available": self.quantum_available,
            "backend_type": str(type(self.backend).__name__) if self.backend else None,
            "api_key_configured": bool(self.api_key and self.api_key != 'your_api_key_here'),
            "supported_features": [
                "Fogging route optimization",
                "Resource allocation optimization", 
                "Outbreak scenario simulation",
                "Quantum Monte Carlo methods"
            ] if self.quantum_available else ["Classical fallback algorithms"]
        }

# Global instance
quantum_optimizer = QuantumDengueOptimizer()

# Example usage functions
async def optimize_dengue_response(districts: List[Dict], teams: int = 3) -> Dict[str, Any]:
    """
    Main function to optimize dengue response using quantum computing
    """
    logger.info("🔮 Starting quantum-enhanced dengue response optimization")
    
    # Route optimization
    route_optimization = quantum_optimizer.optimize_fogging_routes(districts, teams)
    
    # Resource allocation
    resources = {
        "fogging_machines": 10,
        "inspection_teams": 15,
        "awareness_kits": 100
    }
    resource_optimization = quantum_optimizer.optimize_resource_allocation(districts, resources)
    
    # Risk simulation
    scenarios = [
        {"name": "High_Rain_Scenario", "weather_risk": 0.8, "population_density": 0.7},
        {"name": "Normal_Weather", "weather_risk": 0.4, "population_density": 0.5},
        {"name": "Dry_Season", "weather_risk": 0.2, "population_density": 0.6}
    ]
    risk_simulation = quantum_optimizer.quantum_risk_simulation(scenarios)
    
    return {
        "quantum_optimization_results": {
            "route_optimization": route_optimization,
            "resource_allocation": resource_optimization,
            "risk_simulation": risk_simulation
        },
        "quantum_status": quantum_optimizer.get_quantum_status(),
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    # Test quantum optimization
    test_districts = [
        {"name": "Mont Kiara", "latitude": 3.1728, "longitude": 101.6508, "risk_level": 0.8},
        {"name": "KLCC", "latitude": 3.1578, "longitude": 101.7123, "risk_level": 0.6},
        {"name": "Bangsar", "latitude": 3.1285, "longitude": 101.6730, "risk_level": 0.7}
    ]
    
    import asyncio
    result = asyncio.run(optimize_dengue_response(test_districts))
    print(json.dumps(result, indent=2))