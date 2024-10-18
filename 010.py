import numpy as np

def convert_reaction(reaction, compound_data):
    # Split the reaction into reactants and products
    reactants, products = reaction.split("->")
    
    # Convert reactants and products into lists of dictionaries
    reactants_list = [compound_data[compound] for compound in reactants.split("+")]
    products_list = [compound_data[compound] for compound in products.split("+")]
    
    # Combine reactants and products into the desired format
    return [reactants_list, products_list]

def convert_to_string(reaction_data, x, compound_data):
    # Create a reverse mapping from element composition to compound name
    reverse_compound_data = {frozenset(v.items()): k for k, v in compound_data.items()}
    
    # Extract reactants and products with their coefficients
    reactants, products = reaction_data
    reactant_coeffs = x[:len(reactants)]
    product_coeffs = x[len(reactants):]
    
    # Build the string for reactants
    reactant_strs = [f"{coeff} {reverse_compound_data[frozenset(compound.items())]}"
                     for coeff, compound in zip(reactant_coeffs, reactants)]
    
    # Build the string for products
    product_strs = [f"{coeff} {reverse_compound_data[frozenset(compound.items())]}"
                    for coeff, compound in zip(product_coeffs, products)]
    
    # Combine into the final reaction string
    reaction_string = " + ".join(reactant_strs) + " -> " + " + ".join(product_strs)
    
    return reaction_string.replace("1 ", "")

# Example usage
compound_data = {
    "C3H8": {"C": 3, "H": 8},
    "O2": {"O": 2},
    "CO2": {"C": 1, "O": 2},
    "H2O": {"H": 2, "O": 1}
}

#reaction = [[{"C": 3, "H": 8}, {"O": 2}], [{"C": 1, "O": 2}, {"H": 2, "O": 1}]]
reaction = "C3H8+O2->CO2+H2O"



periodic_table = ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne"]
def balance(reaction):
  
  for i in range(2):
    for key in compound_data.keys():
      if compound_data[key] == reaction[i]:
        reaction
  element_present = [[],[]]
  for i in range(2):
    for item in reaction[i]:
      element_present += list(item.keys())
  matrix = []
  for element in periodic_table:
    if element not in element_present:
      continue
    matrix.append([0]*(len(reaction[0])+len(reaction[1])))
    for i in range(2):
      for index, compound in enumerate(reaction[i]):
        if element not in compound.keys():
          continue  
        if i == 0:
          matrix[-1][index] = compound[element]
        else:
          matrix[-1][index+len(reaction[0])] = -compound[element]
  
  B = [0]*len(matrix)+[1]
  matrix.append([0]*(len(reaction[0])+len(reaction[1])))
  matrix[-1][0] = 1
  A = np.array(matrix)
  B = np.array(B)
  return [int(x) for x in np.linalg.solve(A, B)]
tmp = convert_reaction(reaction,compound_data)
print(tmp)
print(reaction)
print(convert_to_string(tmp, balance(tmp), compound_data))

atomic_masses = {
    "H": 1.008,   # Hydrogen
    "He": 4.0026, # Helium
    "Li": 6.94,   # Lithium
    "Be": 9.0122, # Beryllium
    "B": 10.81,   # Boron
    "C": 12.011,  # Carbon
    "N": 14.007,  # Nitrogen
    "O": 15.999,  # Oxygen
    "F": 18.998,  # Fluorine
    "Ne": 20.180  # Neon
}

def calculate_molecular_mass(compound):
    """Calculate the molecular mass of a given compound."""
    atom_counts = compound_data.get(compound)
    
    if atom_counts is None:
        raise ValueError(f"Compound '{compound}' not found in compound data.")
    
    total_mass = 0
    for element, count in atom_counts.items():
        if element in atomic_masses:
            total_mass += atomic_masses[element] * count
        else:
            raise ValueError(f"Element '{element}' not found in atomic masses.")
    
    return total_mass

molecule = "C3H8"  # Propane
molecular_mass = calculate_molecular_mass(molecule)
print(f"The molecular mass of {molecule} is: {molecular_mass:.3f} g/mol")

import sympy as sp

# Initialize variables with some values
variables = {
    "mass_of_solution": None,  # in grams
    "volume_of_solution": None,  # in liters
    "molarity_of_solution": None,  # M
    "molality_of_solution": 5.2,  # Given
    "mole_fraction_of_solute": None,
    "mole_fraction_of_solvent": None,
    "weight_by_weight_ratio": None,
    "weight_by_volume_ratio": None,
    "density_of_solution": None,
    "mass_of_solute": None,  # Example value in grams
    "mass_of_solvent": None,  # in grams
    "mole_of_solute": None,
    "mole_of_solvent": None,
    "atomic_mass_solute": 32,  # Example for solute (g/mol)
    "atomic_mass_solvent": 18  # Example for solvent (g/mol)
}
print(variables.keys())

# Create symbolic variables
mass_of_solution = sp.Symbol('mass_of_solution')  # in grams
volume_of_solution = sp.Symbol('volume_of_solution')  # in liters
molarity_of_solution = sp.Symbol('molarity_of_solution')  # M
molality_of_solution = sp.Symbol('molality_of_solution')  # Given
mole_fraction_of_solute = sp.Symbol('mole_fraction_of_solute')
mole_fraction_of_solvent = sp.Symbol('mole_fraction_of_solvent')
weight_by_weight_ratio = sp.Symbol('weight_by_weight_ratio')
weight_by_volume_ratio = sp.Symbol('weight_by_volume_ratio')
density_of_solution = sp.Symbol('density_of_solution')
mass_of_solute = sp.Symbol('mass_of_solute')  # Example value in grams
mass_of_solvent = sp.Symbol('mass_of_solvent')  # in grams
mole_of_solute = sp.Symbol('mole_of_solute')
mole_of_solvent = sp.Symbol('mole_of_solvent')
atomic_mass_solute = sp.Symbol('atomic_mass_solute')  # Example for solute (g/mol)
atomic_mass_solvent = sp.Symbol('atomic_mass_solvent')  # Example for solvent (g/mol)

# Initialize a list to hold equations
equations = []

# Iterate over the dictionary and create equations for non-None values
for key, value in variables.items():
    if value is not None:
        # Create a corresponding symbolic variable
        sym_var = sp.Symbol(key)
        # Add equation to the list
        equations.append(sp.Eq(sym_var, value))

# Molarity equation: M = (mass_of_solute / atomic_mass_solute) / volume_of_solution
equations.append(sp.Eq(molarity_of_solution, (mass_of_solute / atomic_mass_solute) / volume_of_solution))

# Molality equation: m = mole_of_solute / (mass_of_solvent / 1000)
equations.append(sp.Eq(molality_of_solution, mole_of_solute / (mass_of_solvent / 1000)))

# Mole Fraction equations
equations.append(sp.Eq(mole_fraction_of_solute, mole_of_solute / (mole_of_solute + mole_of_solvent)))
equations.append(sp.Eq(mole_fraction_of_solvent, mole_of_solvent / (mole_of_solute + mole_of_solvent)))

# Weight by Weight Ratio equation: %w/w = (mass_of_solute / mass_of_solution) * 100
equations.append(sp.Eq(weight_by_weight_ratio, (mass_of_solute / mass_of_solution) * 100))

# Weight by Volume Ratio equation: %w/v = (mass_of_solute / (volume_of_solution * 1000))  # g/mL
equations.append(sp.Eq(weight_by_volume_ratio, mass_of_solute / (volume_of_solution * 1000)))

# Density equation: density = mass_of_solution / volume_of_solution
equations.append(sp.Eq(density_of_solution, mass_of_solution / volume_of_solution))

# Convert moles from mass: mole_of_solute = mass_of_solute / atomic_mass_solute
equations.append(sp.Eq(mole_of_solute, mass_of_solute / atomic_mass_solute))

# Convert moles from mass: mole_of_solvent = mass_of_solvent / atomic_mass_solvent
equations.append(sp.Eq(mole_of_solvent, mass_of_solvent / atomic_mass_solvent))

# Solve the equations
solution = sp.solve(equations)

# Print the equations and the solution
print("Equations:")
for eq in equations:
    print(eq)

print("\nSolution:")
for item in solution[0].items():
  print(str(item[0]) + " : " + str(item[1]))
  
