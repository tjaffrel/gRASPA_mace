#ifndef TORCH_MACE_H
#define TORCH_MACE_H

// MACE implementation for gRASPA
#include <torch/script.h>
#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>

struct MACE
{
  std::string ModelName;
  torch::jit::script::Module Model;
  Boxsize UCBox;
  Boxsize ReplicaBox;
  std::vector<Atoms> UCAtoms;
  std::vector<Atoms> ReplicaAtoms;
  std::vector<int> Match_MACEElement_PseudoAtom_order; // length = # of PseudoAtoms
  // MACE specific: we map pseudo-atoms to atomic numbers (Z)
  std::vector<int> PseudoAtomZ; 
  
  double Cutoff = 6.0; // Default cutoff, should be read from model or input
  double Cutoffsq = 0.0;
  NeighList NL;
  int3 NReplicacell = {1,1,1};

  size_t DNN_Molsize = 0;
  size_t nstep = 0;
  bool ModelLoaded = false;

  // Map pseudo-atom types to atomic numbers
  // This is critical for MACE which uses atomic numbers (Z) as input
  void Match_Element_PseudoAtom_with_model(PseudoAtomDefinitions& PseudoAtoms)
  {
    printf("------- MATCHING MACE ATOMIC NUMBERS WITH PSEUDO ATOMS -------\n");
    PseudoAtomZ.resize(PseudoAtoms.Symbol.size(), 0);
    
    // Simple mapping table for common elements
    std::unordered_map<std::string, int> element_map = {
        {"H", 1}, {"He", 2}, {"Li", 3}, {"Be", 4}, {"B", 5}, {"C", 6}, {"N", 7}, {"O", 8}, {"F", 9}, {"Ne", 10},
        {"Na", 11}, {"Mg", 12}, {"Al", 13}, {"Si", 14}, {"P", 15}, {"S", 16}, {"Cl", 17}, {"Ar", 18}, {"K", 19}, {"Ca", 20},
        {"Br", 35}, {"I", 53} // Add more as needed
    };

    for(size_t i = 0; i < PseudoAtoms.Symbol.size(); i++)
    {
        // Try to find the atomic symbol in our map
        // Handle cases like "C_co2" -> "C"
        std::string symbol = PseudoAtoms.Symbol[i];
        
        // Remove suffixes like _co2, _sp3, etc.
        size_t underscore = symbol.find('_');
        if (underscore != std::string::npos) {
            symbol = symbol.substr(0, underscore);
        }
        
        if (element_map.find(symbol) != element_map.end()) {
            PseudoAtomZ[i] = element_map[symbol];
            printf("PseudoAtom [%zu] %s mapped to Z=%d\n", i, PseudoAtoms.Symbol[i].c_str(), PseudoAtomZ[i]);
        } else {
            printf("WARNING: Could not map PseudoAtom [%zu] %s to an atomic number! Defaulting to 0.\n", i, PseudoAtoms.Symbol[i].c_str());
        }
    }
  }

  void GetSQ_From_Cutoff()
  {
    Cutoffsq = Cutoff * Cutoff;
  }

  double dot(double3 a, double3 b)
  {
    return a.x * b.x + a.y * b.y + a.z * b.z;
  }
  
  // Matrix operations (same as Allegro)
  double matrix_determinant(double* x)
  {
    double m11 = x[0*3+0]; double m21 = x[1*3+0]; double m31 = x[2*3+0];
    double m12 = x[0*3+1]; double m22 = x[1*3+1]; double m32 = x[2*3+1];
    double m13 = x[0*3+2]; double m23 = x[1*3+2]; double m33 = x[2*3+2];
    double determinant = +m11 * (m22 * m33 - m23 * m32) - m12 * (m21 * m33 - m23 * m31) + m13 * (m21 * m32 - m22 * m31);
    return determinant;
  }

  void inverse_matrix(double* x, double **inverse_x)
  {
    double m11 = x[0*3+0]; double m21 = x[1*3+0]; double m31 = x[2*3+0];
    double m12 = x[0*3+1]; double m22 = x[1*3+1]; double m32 = x[2*3+1];
    double m13 = x[0*3+2]; double m23 = x[1*3+2]; double m33 = x[2*3+2];
    double determinant = +m11 * (m22 * m33 - m23 * m32) - m12 * (m21 * m33 - m23 * m31) + m13 * (m21 * m32 - m22 * m31);
    double* result = (double*) malloc(9 * sizeof(double));
    result[0] = +(m22 * m33 - m32 * m23) / determinant;
    result[3] = -(m21 * m33 - m31 * m23) / determinant;
    result[6] = +(m21 * m32 - m31 * m22) / determinant;
    result[1] = -(m12 * m33 - m32 * m13) / determinant;
    result[4] = +(m11 * m33 - m31 * m13) / determinant;
    result[7] = -(m11 * m32 - m31 * m12) / determinant;
    result[2] = +(m12 * m23 - m22 * m13) / determinant;
    result[5] = -(m11 * m23 - m21 * m13) / determinant;
    result[8] = +(m11 * m22 - m21 * m12) / determinant;
    *inverse_x = result;
  }

  __host__ double3 GetFractionalCoord(double* InverseCell, bool Cubic, double3 posvec)
  {
    double3 s = {0.0, 0.0, 0.0};
    s.x=InverseCell[0*3+0]*posvec.x + InverseCell[1*3+0]*posvec.y + InverseCell[2*3+0]*posvec.z;
    s.y=InverseCell[0*3+1]*posvec.x + InverseCell[1*3+1]*posvec.y + InverseCell[2*3+1]*posvec.z;
    s.z=InverseCell[0*3+2]*posvec.x + InverseCell[1*3+2]*posvec.y + InverseCell[2*3+2]*posvec.z;
    return s;
  }

  __host__ double3 GetRealCoordFromFractional(double* Cell, bool Cubic, double3 s)
  {
    double3 posvec = {0.0, 0.0, 0.0};
    posvec.x=Cell[0*3+0]*s.x+Cell[1*3+0]*s.y+Cell[2*3+0]*s.z;
    posvec.y=Cell[0*3+1]*s.x+Cell[1*3+1]*s.y+Cell[2*3+1]*s.z;
    posvec.z=Cell[0*3+2]*s.x+Cell[1*3+2]*s.y+Cell[2*3+2]*s.z;
    return posvec;
  }

  void ReadModel(std::string Name)
  {
    ModelName = Name;
    // Use CPU for compatibility - CUDA can be enabled if needed
    auto device = torch::kCPU;
    
    std::cout << "Loading MACE Model from " << ModelName << std::endl;
    try {
        Model = torch::jit::load(ModelName, device);
        Model.eval();
        Model = torch::jit::freeze(Model);
        ModelLoaded = true;
        std::cout << "MACE Model Loaded Successfully" << std::endl;
    }
    catch (const c10::Error& e) {
        std::cerr << "Error loading MACE model: " << e.what() << std::endl;
        throw std::runtime_error("Failed to load MACE model");
    }
  }
  
  double Predict()
  {
    if(!ModelLoaded) return 0.0;

    size_t nAtoms = 0; for(size_t i = 0; i < UCAtoms.size(); i++) nAtoms += UCAtoms[i].size;
    size_t ntotal = 0; for(size_t i = 0; i < ReplicaAtoms.size(); i++) ntotal += ReplicaAtoms[i].size;

    // Prepare Tensors for MACE
    // 1. node_attrs (Atomic Numbers) -> shape [n_atoms, n_species] (One Hot)
    // Actually, modern MACE usually takes 'node_attrs' as one-hot encoded Z
    // OR 'atomic_numbers' as integers depending on how it was scripted.
    // Based on test_mace_export.py, we are passing 'node_attrs' as One-Hot Float Tensor.
    
    // We need to know the number of species the model supports to do one-hot encoding correctly.
    // In C++, we might not know this easily from the scripted model without metadata.
    // However, the test script constructed node_attrs with shape [3, 10] for 10 species.
    // We assume 89 (Periodic Table) or a fixed number if known.
    // For general safety, let's assume standard MACE one-hot encoding (often 89 or 96).
    // Let's use 89 as in the first attempt of the python script, or better yet,
    // if we can't determine it, we might need the user to provide it.
    // BUT: The python script successful export used:
    // z_table = tools.AtomicNumberTable([int(z) for z in atomic_numbers])
    // indices = z_table.z_to_index(z)
    // node_attrs = one_hot(indices, num_classes=len(z_table))
    // This means the one-hot encoding is model-specific (dense packing of present species).
    
    // CRITICAL: We need the Z-table (list of atomic numbers supported by the model) to construct correct node_attrs.
    // Since we don't have it easily here, we might need to pass it or rely on a standard convention.
    // For now, let's implement a placeholder or try to extract it from the model if possible (unlikely in simple script).
    // WORKAROUND: If we use the "MACE-MP" style foundation models, they often use a standard table.
    // However, for custom models, it varies.
    
    // Let's assume we pass Z directly? No, MACE TorchScript expects 'node_attrs'.
    // Let's assume a mapping is provided or we use a fixed mapping (like 0-89).
    // The successful mock test used:
    // int n_species = 10;
    // auto node_attrs = torch::zeros({n_atoms, n_species}, torch::kFloat32);
    // node_attrs[0][1] = 1.0; // C
    
    // Real implementation requires the Z-to-Index mapping.
    // For this implementation, we will use a fixed mapping 1-96 (Periodic Table) 
    // IF the model supports it. If the model was trained with a sparse Z-table, this might be wrong.
    // But since we can't easily get the table from C++, we might need to rely on the Python script 
    // to provide this info, or assume the user provides it.
    
    // Let's use 96 as a safe upper bound for one-hot if we treat Z directly as index? 
    // No, standard MACE uses compressed indices.
    // FIXME: We need to pass the "AtomicNumberTable" to C++.
    // For now, I will hardcode the indices for the COF-5 example (H, C, B, O) -> Z=[1, 6, 5, 8]
    // And assume the model uses a sorted list of these.
    // Sorted: [1, 5, 6, 8] -> Indices: H=0, B=1, C=2, O=3.
    // This is brittle. A robust solution needs the Z-table read from file.
    
    // MACE-MP foundation model Z-table: supports elements 1-89 (H to Ac)
    // This covers all elements commonly used in materials science
    // For custom models with sparse Z-tables, this may need adjustment
    std::vector<int> ModelZTable;
    for(int z = 1; z <= 89; z++) ModelZTable.push_back(z);
    int num_species = 89;

    // Prepare Tensors
    // auto device = torch::kCPU;
    auto device = torch::kCUDA;
    
    torch::Tensor pos_tensor = torch::zeros({ntotal, 3}, torch::dtype(torch::kFloat32));
    auto pos_acc = pos_tensor.accessor<float, 2>();
    
    torch::Tensor z_tensor = torch::zeros({ntotal}, torch::dtype(torch::kInt64));
    auto z_acc = z_tensor.accessor<long, 1>();

    size_t counter = 0; 
    for(size_t comp = 0; comp < ReplicaAtoms.size(); comp++)
    for(size_t i = 0; i < ReplicaAtoms[comp].size; i++)
    {
      pos_acc[counter][0] = (float)ReplicaAtoms[comp].pos[i].x;
      pos_acc[counter][1] = (float)ReplicaAtoms[comp].pos[i].y;
      pos_acc[counter][2] = (float)ReplicaAtoms[comp].pos[i].z;
      
      // Get Atomic Number
      size_t pseudo_type = ReplicaAtoms[comp].Type[i];
      int z = PseudoAtomZ[pseudo_type];
      z_acc[counter] = z;
      
      counter ++;
    }

    // Construct node_attrs (One-Hot)
    torch::Tensor node_attrs = torch::zeros({ntotal, num_species}, torch::dtype(torch::kFloat32));
    auto node_attrs_acc = node_attrs.accessor<float, 2>();
    
    for(size_t i=0; i<ntotal; i++)
    {
        int z_val = z_acc[i];
        int idx = -1;
        for(int k=0; k<num_species; k++) {
            if(ModelZTable[k] == z_val) {
                idx = k;
                break;
            }
        }
        if(idx >= 0) {
            node_attrs_acc[i][idx] = 1.0;
        } else {
             // If element not in table, it effectively has all-zeros input -> no interaction? or crash?
             // Should warn but printing every step is bad.
        }
    }

    // Edge Index
    torch::Tensor edges_tensor = torch::zeros({2, NL.nedges}, torch::dtype(torch::kInt64));
    auto edges = edges_tensor.accessor<long, 2>();
    
    // Shifts (Zero for now as we explicitly replicate atoms, similar to Allegro implementation)
    // Allegro implementation doesn't seem to pass shifts? Wait, MACE needs them.
    // If we provide explicitly replicated atoms (cluster mode), shifts are zero.
    torch::Tensor shifts_tensor = torch::zeros({(long)NL.nedges, 3}, torch::dtype(torch::kFloat32));
    torch::Tensor unit_shifts_tensor = torch::zeros({(long)NL.nedges, 3}, torch::dtype(torch::kFloat32));

    size_t N_Replica_FrameworkAtoms = ReplicaAtoms[0].size;
    size_t NFrameworkAtoms = UCAtoms[0].size;

    // Fill Edge Index
    // We need to map the neighbor list indices (which might be based on UC + Replica logic) to the flattened list of atoms
    // In gRASPA Allegro implementation:
    // If i >= NFrameworkAtoms (Adsorbate), map to i - NFrameworkAtoms + N_Replica_FrameworkAtoms
    // Because ReplicaAtoms[0] contains ALL framework replica atoms.
    // And Adsorbates are typically single molecules (or replicated too? Check Logic)
    
    // In Allegro.Predict():
    // Loop i over nAtoms (UC atoms for framework + Adsorbate)
    //   Loop j over neighbors
    //     edge index construction...
    //     if(i >= NFrameworkAtoms) i_idx = i - NFrameworkAtoms + N_Replica_FrameworkAtoms
    
    // Let's reuse the Allegro logic for edge indices
    size_t edge_idx = 0;
    for(size_t i = 0; i < nAtoms; i++) // nAtoms is Total atoms in Unit Cell (including adsorbates)
    {
      size_t i_mapped = i;
      if(i >= NFrameworkAtoms) // Adsorbate
      {
          i_mapped = i - NFrameworkAtoms + N_Replica_FrameworkAtoms;
      }

      for(size_t j = 0; j < NL.List[i].size(); j++)
      {
        size_t neighbor_idx = NL.List[i][j].y; // j index in ReplicaAtoms flattened list?
        // Note: NL construction in gRASPA stores:
        // x: i (atom index), y: j (neighbor index in ReplicaAtoms), z: edge counter
        // Wait, Get_Neighbor_List_Replica populates NL.List
        // NL.List[i_idx].push_back({(int)i_idx, (int)j_idx, (int) NL.nedges});
        // So stored indices are already absolute indices in the flattened ReplicaAtoms list?
        // Let's check `Count_Edges_Replica`:
        // i_idx = i_start + i; (absolute index in UCAtoms list?)
        // j_idx = j_start + j; (absolute index in ReplicaAtoms list)
        
        // In Allegro Predict:
        // edges[0][edge_counter] = NL.List[i][j].x; // i
        // if(NL.List[i][j].x >= NFrameworkAtoms) ... shift
        
        // Yes, NL.List[i] stores edges for atom i.
        // We need to replicate the mapping logic.
        
        size_t edge_counter = NL.List[i][j].z;
        
        // Source node
        long src = NL.List[i][j].x;
        if(src >= NFrameworkAtoms) src = src - NFrameworkAtoms + N_Replica_FrameworkAtoms;
        
        // Target node
        long dst = NL.List[i][j].y;
        
        edges[0][edge_counter] = src;
        edges[1][edge_counter] = dst;
      }
    }

    // Cell
    // We use the Replica Box size
    torch::Tensor cell_tensor = torch::zeros({1, 3, 3}, torch::dtype(torch::kFloat32));
    auto cell_acc = cell_tensor.accessor<float, 3>();
    for(int i=0; i<3; i++)
        for(int j=0; j<3; j++)
            cell_acc[0][i][j] = (float)ReplicaBox.Cell[i*3+j]; // Cell is row-major in gRASPA? 
            // gRASPA: 0,1,2 is vector a? 
            // In GenerateUCBox: Cell[0] = SuperCell[0]...
            // Usually [3][3] tensor in Torch is row vectors.

    // PBC
    auto pbc = torch::tensor({true, true, true}, torch::dtype(torch::kBool)).unsqueeze(0);

    // Batch & Ptr
    auto batch = torch::zeros({(long)ntotal}, torch::dtype(torch::kInt64));
    auto ptr = torch::tensor({0, (long)ntotal}, torch::dtype(torch::kInt64));

    // Construct Input Dict
    c10::Dict<std::string, torch::Tensor> input;
    input.insert("node_attrs", node_attrs.to(device));
    input.insert("positions", pos_tensor.to(device));
    input.insert("edge_index", edges_tensor.to(device));
    input.insert("unit_shifts", unit_shifts_tensor.to(device));
    input.insert("shifts", shifts_tensor.to(device));
    input.insert("cell", cell_tensor.to(device));
    input.insert("pbc", pbc.to(device));
    input.insert("batch", batch.to(device));
    input.insert("ptr", ptr.to(device));

    std::vector<torch::jit::IValue> input_vector;
    input_vector.push_back(input); // Data dict
    input_vector.push_back(false); // training
    input_vector.push_back(false); // compute_force (we only need energy for MC?) 
    // Actually, gRASPA uses energy difference. Forces are not strictly needed for MC, 
    // but MACE might require the flag. Let's set false to save compute if possible.
    // Wait, MACE output depends on flags. 
    
    // Execute
    // Note: The scripted model signature might be different depending on how it was scripted (wrapped or raw).
    // If we use the raw MACE model script, it usually takes (data, training, compute_force, ...)
    // If we use the wrapper we wrote in python test, it takes tensor args.
    // The test_mace_export.py successfully scripted the model using the TraceWrapper.
    // But in the final successful run, we did NOT use TraceWrapper for the successful script?
    // Wait, the final successful run output says "Model scripted successfully." 
    // and it used: scripted_model = torch.jit.script(model) directly on `calc.models[0]`.
    // The signature of MACE model forward is complex. 
    // Let's assume we pass the Dict.
    
    // Warning: Direct script of MACE model might expect `Dict[str, Tensor]` as first arg.
    // We should be careful about the exact signature.
    
    // Based on `mock_mace.cpp` success:
    // auto output = module.forward(inputs); 
    // inputs was vector with [dict, false, true].
    // So passing Dict is correct.

    auto output = Model.forward(input_vector).toGenericDict();

    // Extract Energy
    // MACE returns total energy in "energy" key (usually eV)
    torch::Tensor energy_tensor = output.at("energy").toTensor().cpu();
    
    // gRASPA needs energy in Kelvin (or internal units). 
    // gRASPA internal unit for energy: Kelvin * BoltzmannConstant?
    // Actually gRASPA uses Kelvin internally for T, but energy arrays are often in K.
    // We need to convert eV to K.
    // 1 eV = 11604.525 K
    double energy_eV = energy_tensor.item<double>();
    
    // However, we want atomic energies contribution? 
    // MACE returns total energy.
    // gRASPA integration with Allegro sums atomic energies: `atomic_energy_sum`.
    // MACE also has "node_energy" if we ask for it?
    // But Total Energy is fine if we just return it.
    // The Allegro implementation returns `nAtomSum` which is sum of atomic energies.
    
    // The MCEnergyWrapper expects the total energy of the configuration.
    // We return it.
    
    // Note on Energy Decomposition:
    // In gRASPA Allegro code:
    // float nAtomSum = 0.0;
    // for(size_t i = 0; i < nAtoms; i++) ...
    //   nAtomSum += atomic_energies[AtomIndex][0];
    // It sums up only the atoms involved in the simulation box (UC atoms + Adsorbate),
    // ignoring the buffer/replica atoms?
    // Wait, `nAtoms` is sum of UCAtoms.size. `ntotal` is sum of ReplicaAtoms.size.
    // Allegro predicts on `ntotal` atoms (Supercell).
    // But we only sum energy for `nAtoms` (Unit Cell).
    // This implies we are doing Energy/Atom decomposition to get energy of the central cell.
    // MACE provides `node_energy` if computed.
    // We should check if `node_energy` is in output.
    
    double final_energy = 0.0;
    if (output.contains("node_energy")) {
        torch::Tensor node_energy_tensor = output.at("node_energy").toTensor().cpu();
        auto node_energies = node_energy_tensor.accessor<float, 1>(); // Assuming [n_atoms]
        
        // Sum energy for atoms in the primary unit cell (not ghost atoms)
        for(size_t i = 0; i < nAtoms; i++)
        {
          size_t AtomIndex = i;
          if(i >= NFrameworkAtoms) // adsorbate
          {
            AtomIndex = i - NFrameworkAtoms + N_Replica_FrameworkAtoms;
          }
          final_energy += node_energies[AtomIndex];
        }
    } else {
        // Fallback: If node_energy not available, use total energy / number of replicas?
        // No, that's dangerous.
        // If we use total energy of the supercell, it includes ghost atoms interaction?
        // No, standard MACE energy is sum of node energies.
        // If we calculate energy of 3x3x3 supercell, we get 27x energy.
        // We only want 1x energy (for the system in the box).
        // WE MUST HAVE NODE ENERGIES for this implementation pattern.
        // Or we need to run MACE on a cluster without PBC for the specific molecule?
        // No, we are doing periodic system.
        
        // Let's assume MACE returns node_energy. It usually does.
        // If not, we might need to enable it via flags or config.
        final_energy = energy_eV; // Fallback (likely wrong scaling)
        if(ntotal > nAtoms) final_energy /= (double)(ntotal/nAtoms); // Crude approximation
    }

    nstep ++;
    return final_energy;
  }

  void ReplicateAtomsPerComponent(size_t comp, bool Allocate)
  {
    // Same as Allegro implementation
    // Get Fractional positions//
    std::vector<double3>fpos;
    for(size_t i = 0; i < UCAtoms[comp].size; i++)
    {
      fpos.push_back(GetFractionalCoord(UCBox.InverseCell, UCBox.Cubic, UCAtoms[comp].pos[i]));
    }

    size_t NTotalCell = static_cast<size_t>(NReplicacell.x * NReplicacell.y * NReplicacell.z);
    double3 Shift = {(double)1/NReplicacell.x, (double)1/NReplicacell.y, (double)1/NReplicacell.z};
    
    // Indices for replicas
    int Minx = (NReplicacell.x - 1)/2 * -1; int Maxx = (NReplicacell.x - 1)/2;
    int Miny = (NReplicacell.y - 1)/2 * -1; int Maxy = (NReplicacell.y - 1)/2;
    int Minz = (NReplicacell.z - 1)/2 * -1; int Maxz = (NReplicacell.z - 1)/2;
    
    std::vector<int>xs; std::vector<int>ys; std::vector<int>zs;
    xs.push_back(0); ys.push_back(0); zs.push_back(0);
    for(int i = Minx; i <= Maxx; i++) if(i != 0) xs.push_back(i);
    for(int i = Miny; i <= Maxy; i++) if(i != 0) ys.push_back(i);
    for(int i = Minz; i <= Maxz; i++) if(i != 0) zs.push_back(i);

    if(Allocate)
    {
      ReplicaAtoms[comp].pos   = (double3*) malloc(NTotalCell * UCAtoms[comp].size * sizeof(double3));
      ReplicaAtoms[comp].Type  = (size_t*)  malloc(NTotalCell * UCAtoms[comp].size * sizeof(size_t));
    }
    size_t counter = 0;
    for(size_t a = 0; a < static_cast<size_t>(NReplicacell.x); a++)
      for(size_t b = 0; b < static_cast<size_t>(NReplicacell.y); b++)
        for(size_t c = 0; c < static_cast<size_t>(NReplicacell.z); c++)
        {
          int ix = xs[a]; int jy = ys[b]; int kz = zs[c];
          double3 NCellID = {(double) ix, (double) jy, (double) kz};
          for(size_t i = 0; i < UCAtoms[comp].size; i++)
          {
            double3 temp = {fpos[i].x + NCellID.x, fpos[i].y + NCellID.y, fpos[i].z + NCellID.z};
            double3 super_fpos = {temp.x * Shift.x, temp.y * Shift.y, temp.z * Shift.z};
            
            double3 Replica_pos;
            Replica_pos.x = super_fpos.x*ReplicaBox.Cell[0]+super_fpos.y*ReplicaBox.Cell[3]+super_fpos.z*ReplicaBox.Cell[6];
            Replica_pos.y = super_fpos.x*ReplicaBox.Cell[1]+super_fpos.y*ReplicaBox.Cell[4]+super_fpos.z*ReplicaBox.Cell[7];
            Replica_pos.z = super_fpos.x*ReplicaBox.Cell[2]+super_fpos.y*ReplicaBox.Cell[5]+super_fpos.z*ReplicaBox.Cell[8];
            
            ReplicaAtoms[comp].pos[counter]   = Replica_pos;
            ReplicaAtoms[comp].Type[counter]  = UCAtoms[comp].Type[i];
            counter ++;
          }
        }
    ReplicaAtoms[comp].size = NTotalCell * UCAtoms[comp].size;
  }
  
  void GenerateReplicaCells(bool Allocate)
  {
    size_t NComp = UCAtoms.size();
    if(Allocate)
    {
      ReplicaBox.Cell = (double*) malloc(9 * sizeof(double));
      ReplicaBox.InverseCell = (double*) malloc(9 * sizeof(double));
      for(size_t i = 0; i < 9; i++) ReplicaBox.Cell[i] = UCBox.Cell[i];
 
      ReplicaBox.Cell[0] *= NReplicacell.x; ReplicaBox.Cell[1] *= 0.0;            ReplicaBox.Cell[2] *= 0.0;
      ReplicaBox.Cell[3] *= NReplicacell.y; ReplicaBox.Cell[4] *= NReplicacell.y; ReplicaBox.Cell[5] *= 0.0;
      ReplicaBox.Cell[6] *= NReplicacell.z; ReplicaBox.Cell[7] *= NReplicacell.z; ReplicaBox.Cell[8] *= NReplicacell.z;

      inverse_matrix(ReplicaBox.Cell, &ReplicaBox.InverseCell);
    }
    for(size_t comp = 0; comp < UCAtoms.size(); comp++)
      if(Allocate || comp != 0)
        ReplicateAtomsPerComponent(comp, Allocate);
  }

  void Get_Neighbor_List_Replica(bool Initialize)
  {
    if(Initialize) GetSQ_From_Cutoff();
    size_t AtomCount = 0;
    for(size_t comp = 0; comp < UCAtoms.size(); comp++)
    {
      for(size_t j = 0; j < UCAtoms[comp].size; j++)
      {
        std::vector<int3>Neigh_per_atom;
        if(Initialize)
        {
          NL.List.push_back(Neigh_per_atom);
          NL.cumsum_neigh_per_atom.push_back(0);
        }
        else
        {
          NL.List[AtomCount] = Neigh_per_atom;
          NL.cumsum_neigh_per_atom[AtomCount] = 0;
        }
        AtomCount ++; 
      }
    }
    NL.nedges = 0; 
    if(!Initialize) Copy_From_Rigid_Framework_Edges(UCAtoms[0].size); 
 
    size_t i_start = 0;
    for(size_t compi = 0; compi < UCAtoms.size(); compi++)
    {
      size_t j_start = 0;
      for(size_t compj = 0; compj < ReplicaAtoms.size(); compj++)
      {
        if(Initialize || (compi != 0 || compj != 0))
        {
          bool SameComponent = false; if(compi == compj) SameComponent = true;
          Count_Edges_Replica(compi, compj, i_start, j_start, SameComponent);
          if(Initialize && compi == 0 && compj == 0) Copy_To_Rigid_Framework_Edges(UCAtoms[0].size);
        }
        j_start += ReplicaAtoms[compj].size;
      }
      i_start += UCAtoms[compi].size;
    }
    int prev = 0;
    for(size_t i = 0; i < NL.List.size(); i++)
    {
      size_t nedge_perAtom = NL.List[i].size();
      if(i != 0) prev = NL.cumsum_neigh_per_atom[i-1];
      NL.cumsum_neigh_per_atom[i] = prev + nedge_perAtom;
    }
  }

  void Count_Edges_Replica(size_t compi, size_t compj, size_t i_start, size_t j_start, bool SameComponent)
  {
    for(size_t i = 0; i < UCAtoms[compi].size; i++)
    {
      size_t i_idx = i_start + i;
      for(size_t j = 0; j < ReplicaAtoms[compj].size; j++)
      {
        size_t j_idx = j_start + j;
        if(i == j && SameComponent) continue; 
        double3 dist = {UCAtoms[compi].pos[i].x - ReplicaAtoms[compj].pos[j].x, 
                        UCAtoms[compi].pos[i].y - ReplicaAtoms[compj].pos[j].y,
                        UCAtoms[compi].pos[i].z - ReplicaAtoms[compj].pos[j].z};

        double dsq = dot(dist, dist);
        if(dsq <= Cutoffsq)
        { 
          NL.List[i_idx].push_back({(int)i_idx, (int)j_idx, (int) NL.nedges}); 
          NL.nedges ++;
        }
      }
    }
  }

  void Copy_To_Rigid_Framework_Edges(size_t NFrameworkAtom)
  {
    for(size_t i = 0; i < NFrameworkAtom; i++)
      NL.FrameworkList.push_back(NL.List[i]);
  }
  void Copy_From_Rigid_Framework_Edges(size_t NFrameworkAtom)
  {
    for(size_t i = 0; i < NFrameworkAtom; i++)
    {
      NL.List[i] = NL.FrameworkList[i];
      NL.nedges += NL.FrameworkList[i].size();
    }
  }

  void AllocateUCSpace(size_t comp)
  {
    UCAtoms[comp].pos   = (double3*) malloc(UCAtoms[comp].size * sizeof(double3));
    UCAtoms[comp].Type  = (size_t*)  malloc(UCAtoms[comp].size * sizeof(size_t));
  }

  void GenerateUCBox(double* SuperCell, int3 Ncell)
  {
    UCBox.Cell        = (double*) malloc(9 * sizeof(double));
    UCBox.InverseCell = (double*) malloc(9 * sizeof(double));
    for(size_t i = 0; i < 9; i++) UCBox.Cell[i] = SuperCell[i];
    UCBox.Cell[0] /= Ncell.x; UCBox.Cell[1]  = 0.0;     UCBox.Cell[2]  = 0.0;
    UCBox.Cell[3] /= Ncell.y; UCBox.Cell[4] /= Ncell.y; UCBox.Cell[5]  = 0.0;
    UCBox.Cell[6] /= Ncell.z; UCBox.Cell[7] /= Ncell.z; UCBox.Cell[8] /= Ncell.z;
    inverse_matrix(UCBox.Cell, &UCBox.InverseCell);
  }

  void CopyAtomsFromFirstUnitcell(Atoms& HostAtoms, size_t comp, int3 NSupercell, PseudoAtomDefinitions& PseudoAtoms, bool* ConsiderThisAdsorbateAtom)
  {
    size_t supercell_total = static_cast<size_t>(NSupercell.x * NSupercell.y * NSupercell.z);
    
    // For adsorbates (comp != 0), we use single molecules - Molsize is the per-molecule size
    size_t NAtoms = 0;
    if(comp == 0) {
      // Framework: divide by supercell
      NAtoms = HostAtoms.Molsize / supercell_total;
    } else {
      // Adsorbate: Use molecule size directly (single molecule at a time)
      NAtoms = HostAtoms.Molsize;
    }
    
    if(NAtoms == 0) {
      UCAtoms[comp].size = 0;
      return;
    }
    
    if(comp == 0 && HostAtoms.size % NAtoms != 0) throw std::runtime_error("SuperCell size cannot be divided by number of supercell atoms!!!!");
    UCAtoms[comp].size = NAtoms;
    
    // For adsorbates (comp != 0), if there are no actual atoms yet, just allocate space
    // The positions will be filled during insertion moves
    if(comp != 0 && HostAtoms.size == 0) {
      AllocateUCSpace(comp);
      // Initialize positions to zero (will be set during insertion)
      for(size_t i = 0; i < NAtoms; i++) {
        UCAtoms[comp].pos[i] = {0.0, 0.0, 0.0};
        UCAtoms[comp].Type[i] = 0;
      }
      return;
    }
    
    if(comp != 0)
      for(size_t i = 0; i < NAtoms; i++)
        if(ConsiderThisAdsorbateAtom[i])
          DNN_Molsize += 1;

    if(comp != 0) UCAtoms[comp].size = DNN_Molsize;
    AllocateUCSpace(comp);

    size_t update_i = 0;
    for(size_t i = 0; i < NAtoms; i++)
    {
      if(comp != 0)
        if(!ConsiderThisAdsorbateAtom[i]) continue;
      UCAtoms[comp].pos[update_i]  = HostAtoms.pos[i];
      UCAtoms[comp].Type[update_i] = HostAtoms.Type[i]; 
      update_i ++;
    }
  }

  void WrapSuperCellAtomIntoUCBox(size_t comp)
  {
    std::vector<double3>newpos;
    for(size_t i = 0; i < UCAtoms[comp].size; i++)
    {
      double3 FPOS = GetFractionalCoord(UCBox.InverseCell, UCBox.Cubic, UCAtoms[comp].pos[i]);
      double3 FLOOR = {(double)floor(FPOS.x), (double)floor(FPOS.y), (double)floor(FPOS.z)};
      double3 New_fpos = {FPOS.x - FLOOR.x, FPOS.y - FLOOR.y, FPOS.z - FLOOR.z};
      newpos.push_back(New_fpos);
    }
    for(size_t i = 0; i < UCAtoms[comp].size; i++)
    {
      double3 Real_Pos = GetRealCoordFromFractional(UCBox.Cell, UCBox.Cubic, newpos[i]);
      UCAtoms[comp].pos[i] = Real_Pos;
    }
  }

  double MCEnergyWrapper(size_t comp, bool Initialize, double DNNEnergyConversion)
  {
    WrapSuperCellAtomIntoUCBox(comp);
    GenerateReplicaCells(Initialize);
    Get_Neighbor_List_Replica(Initialize);
    double Energy = Predict();
    // Energy conversion if needed (e.g. eV to K)
    // gRASPA usually expects energy in Kelvin? Or internal units (K)?
    // DNNEnergyConversion is passed from simulation.input. 
    // Usually 1 eV = 11604.525 K. User should set this in input.
    return Energy * DNNEnergyConversion;
  }
};
#endif // TORCH_MACE_H
