#include "../include/CardiacProblem.hpp"
#include <memory> // Inserito per supportare std::shared_ptr

namespace CardiacProject {
    using namespace dealii;

    template <int dim>
    CardiacProblem<dim>::CardiacProblem(MeshMode mode, const std::string &mesh_filename)
        : current_mode(mode),                       
          external_mesh_filename(mesh_filename),    
          mpi_communicator(MPI_COMM_WORLD),
          triangulation(mpi_communicator),
          fe(1), 
          dof_handler(triangulation),
          pcout(std::cout, (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)),
          computing_timer(mpi_communicator, pcout, TimerOutput::summary,
                          TimerOutput::wall_times) 
    {
        // Impostazioni temporali iniziali della simulazione. 
        time = 0.0;
        time_step = 0.001; // ms 
        final_time = 5.0; // ms
    }

    template <int dim>
    void CardiacProblem<dim>::setup_grid() {
        TimerOutput::Scope t(computing_timer, "Setup Grid");

        if (current_mode == MeshMode::ExternalLoad) {
            // ==========================================
            // BINARIO 2: MODALITÀ PRODUZIONE (Mesh Esterna)
            // ==========================================
            pcout << "=> Modalità Produzione: Caricamento mesh ottimizzata dal file " 
                  << external_mesh_filename << " ..." << std::endl;
            
            GridIn<dim> grid_in;
            grid_in.attach_triangulation(triangulation);
            std::ifstream input_file(external_mesh_filename);
            
            AssertThrow(input_file.is_open(), 
                        ExcMessage("Errore: impossibile aprire il file della mesh!"));
            
            grid_in.read_msh(input_file);
            pcout << "Mesh esterna caricata con successo!" << std::endl;

        } else {
            // ==========================================
            // BINARIO 1: MODALITÀ BENCHMARK (Mesh Interna)
            // ==========================================
            pcout << "=> Modalità Benchmark: Generazione interna della mesh in corso..." << std::endl;
            
            // Costruzione della mesh rettangolare che rappresenta il tessuto cardiaco.
            // h controlla la discretizzazione spaziale iniziale.
            double h = 0.5; // Inizia grossolano (0.5mm), poi raffinabile

            std::vector<unsigned int> repetitions(dim);
            Point<dim> p1, p2;

            // Dimensioni geometriche del dominio in mm.
            p1[0] = 0.0; p1[1] = 0.0; 
            p2[0] = 20.0; p2[1] = 7.0;
            if (dim == 3) {
                p1[2] = 0.0; p2[2] = 3.0;
            }

            for(unsigned int d=0; d<dim; ++d)
                repetitions[d] = static_cast<unsigned int>((p2[d]-p1[d])/h);

            GridGenerator::subdivided_hyper_rectangle(triangulation, repetitions, p1, p2);

            // Tag del materiale: il box vicino all'origine rappresenta la zona di stimolo.
            for (const auto &cell : triangulation.active_cell_iterators()) {
                if (cell->is_locally_owned()) {
                    Point<dim> center = cell->center();
                    bool in_stimulus_zone = true;
                    // L'area di stimolo è definita come un cubo di lato 1.5 mm.
                    for(unsigned int d=0; d<dim; ++d)
                        if(center[d] > 1.5) in_stimulus_zone = false;

                    if (in_stimulus_zone) cell->set_material_id(1); // ID 1 = Stimolo
                    else cell->set_material_id(0); // ID 0 = Tessuto passivo
                }
            }
        }
    }

    template <int dim>
    void CardiacProblem<dim>::setup_system() {
        TimerOutput::Scope t(computing_timer, "Setup System");
            
            // Distribuzione dei DoF sulla mesh parallela.
            dof_handler.distribute_dofs(fe);

            // DoF posseduti localmente e DoF necessari anche come ghost.
            locally_owned_dofs = dof_handler.locally_owned_dofs();
            locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);
            
            // Informazioni diagnostiche sul partizionamento MPI.
            pcout << "Process " << Utilities::MPI::this_mpi_process(mpi_communicator) 
                  << ": owned=" << locally_owned_dofs.n_elements() 
                  << ", relevant=" << locally_relevant_dofs.n_elements()
                  << ", ghost=" << (locally_relevant_dofs.n_elements() - locally_owned_dofs.n_elements())
                  << std::endl;

            // Conteggio globale utile per verificare la discretizzazione.
            pcout << "Number of active cells: " << triangulation.n_global_active_cells() << std::endl;
            pcout << "Number of degrees of freedom: " << dof_handler.n_dofs() << std::endl;

            // Vincoli di continuità per i nodi hanging tra celle di dimensione diversa.
            constraints.clear();
            constraints.reinit(locally_relevant_dofs);
            DoFTools::make_hanging_node_constraints(dof_handler, constraints);
            constraints.close();

            // Costruzione della sparsity pattern per le matrici globali.
            DynamicSparsityPattern dsp(locally_relevant_dofs);
            DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
            SparsityPattern sparsity_pattern;
            sparsity_pattern.copy_from(dsp);

            // Allocazione delle strutture lineari distribuite.
            system_matrix.reinit(locally_owned_dofs, locally_owned_dofs, sparsity_pattern, mpi_communicator);
            mass_matrix.reinit(locally_owned_dofs, locally_owned_dofs, sparsity_pattern, mpi_communicator);
            // solution contiene i DoF posseduti dal processo corrente.
            solution.reinit(locally_owned_dofs, mpi_communicator);
            // locally_relevant_solution include anche i DoF ghost per letture in assembly.
            locally_relevant_solution.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
            system_rhs.reinit(locally_owned_dofs, mpi_communicator);

            // Inizializzazione dei gate ionici locali, uno per ogni DoF owned.
            unsigned int n_local_dofs = locally_owned_dofs.n_elements();
            gate_v.resize(n_local_dofs, 1.0); // v inizia a 1 (riposo)
            gate_w.resize(n_local_dofs, 1.0); // w inizia a 1
            gate_s.resize(n_local_dofs, 0.0); // s inizia a 0

            // Tempo di attivazione per ogni DoF: -1 significa non ancora attivato.
            activation_time.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
            activation_time = -1.0; // Inizializza a -1 (non attivato)
    }

    template <int dim>
    void CardiacProblem<dim>::assemble_system()
    {
        TimerOutput::Scope t(computing_timer, "Assembly");

        // -------------------------------------------------------------
        // Lucchetto statico per bloccare il ricalcolo.
        // -------------------------------------------------------------
        static bool matrix_is_assembled = false;

        // Azzeriamo la gigantesca matrice SOLO al primissimo giro.
        if (!matrix_is_assembled) {
            system_matrix = 0.0;
        }
        
        // Il vettore va sempre azzerato perché la cronologia cambia.
        system_rhs = 0.0;
        
        // Conteggio locale delle celle effettivamente assemblate dal processo.
        unsigned int n_cells_assembled = 0;

        QGauss<dim> quadrature_formula(fe.degree + 1);
        FEValues<dim> fe_values(fe,
                                quadrature_formula,
                                update_values | update_gradients |
                                update_quadrature_points | update_JxW_values);

        const unsigned int dofs_per_cell = fe.dofs_per_cell;
        const unsigned int n_q_points = quadrature_formula.size();

        FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
        Vector<double> cell_rhs(dofs_per_cell);
        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
        std::vector<double> current_solution_values(n_q_points);

        // Parametri fisici normalizzati del problema di diffusione.
        // La diffusione è anisotropa: lungo fibra e trasversalmente la costante cambia.
        const double diff_l = 0.12; 
        const double diff_t = 0.013; 

        // Termine di stimolo applicato nella zona iniziale e solo nei primi istanti.
        const double Stim_val = 0.5; 

        Tensor<2, dim> diffusion;
        diffusion[0][0] = diff_l;
        diffusion[1][1] = diff_t;
        if (dim == 3) diffusion[2][2] = diff_t;

        // Loop sulle celle locali: ogni cella contribuisce con il suo stencil FEM.
        for (const auto &cell : dof_handler.active_cell_iterators())
        {
            if (cell->is_locally_owned())
            {
                n_cells_assembled++;
                cell_rhs = 0;
                
                if (!matrix_is_assembled) {
                    cell_matrix = 0;
                }
                
                fe_values.reinit(cell);
                fe_values.get_function_values(locally_relevant_solution, current_solution_values);

                for (unsigned int q = 0; q < n_q_points; ++q)
                {
                    const double JxW = fe_values.JxW(q);

                    for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    {
                        const auto phi_i = fe_values.shape_value(i, q);
                        
                        // PARTE 1: LHS (Rigidezza e Massa)
                        // Il lucchetto fa scartare questo pesantissimo calcolo per i restanti N-1 cicli.
                        if (!matrix_is_assembled) {
                            const auto grad_phi_i = fe_values.shape_grad(i, q);
                            for (unsigned int j = 0; j < dofs_per_cell; ++j)
                            {
                                const auto phi_j = fe_values.shape_value(j, q);
                                const auto grad_phi_j = fe_values.shape_grad(j, q);

                                // MASSA: Ora è solo (1/dt) perché abbiamo diviso per Chi*Cm
                                const double mass_term = (1.0 / time_step) * phi_i * phi_j;
                                // STIFFNESS: Usa il tensore di diffusione D
                                const double stiff_term = (grad_phi_i * (diffusion * grad_phi_j));

                                cell_matrix(i, j) += (mass_term + stiff_term) * JxW;
                            }
                        }

                        // PARTE 2: RHS (Aggiornamento termico-cronologico sempre eseguito)
                        // Termine esplicito che riporta la soluzione al passo precedente.
                        double rhs_val = (1.0 / time_step) * current_solution_values[q] * phi_i;

                        // Stimolo geometrico basato sulla posizione del punto di quadratura.
                        Point<dim> q_point = fe_values.quadrature_point(q);
                        bool in_stim = true;
                        for(unsigned int d=0; d<dim; ++d) 
                            if(q_point[d] > 1.5) in_stim = false;

                        if (in_stim && time <= 2.0) { 
                            rhs_val += Stim_val * phi_i; 
                        }

                        cell_rhs(i) += rhs_val * JxW;
                    }
                }
                cell->get_dof_indices(local_dof_indices);
                
                // Distribuzione MPI Dinamica: Invia la matrice solo se è la prima volta.
                if (!matrix_is_assembled) {
                    constraints.distribute_local_to_global(cell_matrix, cell_rhs,
                                                          local_dof_indices,
                                                          system_matrix, system_rhs);
                } else {
                    constraints.distribute_local_to_global(cell_rhs, local_dof_indices, system_rhs);
                }
            }
        }
        
        // Comprimiamo le somme parallele MPI globali
        if (!matrix_is_assembled) {
            system_matrix.compress(VectorOperation::add); // Finalizzazione delle somme parziali parallele.
            matrix_is_assembled = true; // Chiudiamo definitivamente il lucchetto!
            
            if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
                pcout << "=> Matrice di sistema assemblata (Il calcolo non verrà ripetuto)." << std::endl;
        }
        system_rhs.compress(VectorOperation::add); // Finalizzazione delle somme parziali parallele.
        
        // Messaggio diagnostico emesso solo alla prima assemble.
        static bool first_call = true;
        if (first_call) {
            pcout << "Process " << Utilities::MPI::this_mpi_process(mpi_communicator)
                  << " assembled " << n_cells_assembled << " cells" << std::endl;
            first_call = false;
        }
    }

    template <int dim>
    void CardiacProblem<dim>::solve_pde()
    {
        TimerOutput::Scope t(computing_timer, "Solve PDE");

        // Risolutore lineare CG con tolleranza relativa sul residuo.
        SolverControl solver_control(10000, 1e-6 * system_rhs.l2_norm());
        TrilinosWrappers::SolverCG solver(solver_control);
        
        // -------------------------------------------------------------
        // CORREZIONE HPC: Smart Pointer Statico per conservare l'AMG.
        // -------------------------------------------------------------
        // Precondizionatore AMG, adatto a problemi diffusive distribuiti in MPI.
        // Usare un shared_ptr statico permette di inizializzare l'AMG solo al primo ciclo e riutilizzarlo per tutti i successivi, risparmiando tempo prezioso.
        static std::shared_ptr<TrilinosWrappers::PreconditionAMG> preconditioner;
        
        if (!amg_preconditioner) {
            if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
                pcout << "=> Inizializzazione Precondizionatore AMG (Operazione una-tantum)..." << std::endl;

            amg_preconditioner = std::make_shared<TrilinosWrappers::PreconditionAMG>();
            TrilinosWrappers::PreconditionAMG::AdditionalData amg_data;
            amg_preconditioner->initialize(system_matrix, amg_data);
        }
        // Il solver scrive solo nel vettore owned.
        solver.solve(system_matrix, solution, system_rhs, *amg_preconditioner);

        // Applica i vincoli e sincronizza la soluzione tra i processi.
        constraints.distribute(solution);
        solution.compress(VectorOperation::insert);
        
        // Copia verso il vettore ghosted usato nelle letture successive.
        locally_relevant_solution = solution;
    }

    template <int dim>
    void CardiacProblem<dim>::solve_ode()
    {
        TimerOutput::Scope t(computing_timer, "Solve ODE");
                
        // Evoluzione locale della reazione ionica su soli DoF posseduti.
        unsigned int local_idx = 0;
        IndexSet::ElementIterator idx = locally_owned_dofs.begin();
        IndexSet::ElementIterator endIdx = locally_owned_dofs.end();

        for(; idx != endIdx; ++idx, ++local_idx) 
        {
            types::global_dof_index global_i = *idx;
            
            // Copia lo stato corrente dal vettore globale ghosted ai gate locali.
            IonicState state;
            state.u = locally_relevant_solution(global_i);
            state.v = gate_v[local_idx];
            state.w = gate_w[local_idx];
            state.s = gate_s[local_idx];

            // 1. Calcolo della corrente ionica totale.
            double J_ion = ionic_model.compute_total_current(state);

            // Limite di sicurezza per evitare esplosioni numeriche.
            if (std::abs(J_ion) > 100.0) J_ion = (J_ion > 0 ? 100.0 : -100.0);

            // 2. Aggiornamento del potenziale transmembrana.
            state.u -= time_step * J_ion; 
            // Mantiene il potenziale in un intervallo fisicamente coerente.
            state.u = std::max(0.0, std::min(1.0, state.u));

            // 3. Evoluzione dei gate ionici.
            ionic_model.evolve_gates(state, time_step);
            
            // Limita i gate nello stesso intervallo per robustezza numerica.
            state.v = std::max(0.0, std::min(1.0, state.v));
            state.w = std::max(0.0, std::min(1.0, state.w));
            state.s = std::max(0.0, std::min(1.0, state.s));

            // Scrive nel vettore owned, quindi senza conflitti MPI.
            solution(global_i) = state.u;
            gate_v[local_idx] = state.v;
            gate_w[local_idx] = state.w;
            gate_s[local_idx] = state.s;
        }
        
        solution.compress(VectorOperation::insert);
        
        // Aggiorna i ghost dopo la fase ODE.
        locally_relevant_solution = solution;
    }

    template <int dim>
    void CardiacProblem<dim>::run()
    {
        // Messaggio iniziale con il numero di processi MPI attivi.
        pcout << "Running Cardiac Problem on " 
            << Utilities::MPI::n_mpi_processes(mpi_communicator) 
            << " processes." << std::endl;

         // Fase 1: costruzione del dominio e setup del sistema FEM.
        setup_grid();
        setup_system();

        // Condizioni iniziali: potenziale nullo ovunque per partire da uno stato semplice.
        VectorTools::interpolate(dof_handler, Functions::ZeroFunction<dim>(), solution);
        locally_relevant_solution = solution;  // Sincronizza i ghost prima del primo output.

        // Scrive il primo snapshot al tempo t=0.
        output_results(0); 

        // Preparazione del ciclo temporale.
        unsigned int time_step_number = 0;
        const unsigned int total_steps = static_cast<unsigned int>(final_time / time_step);
        unsigned int last_progress = 0;
        
        if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
            std::cout << "Progress: 0%" << std::flush;
        
        while (time < final_time)
        {
            time_step_number++;
            time += time_step;
            
            // 1. Fase diffusiva: costruzione e soluzione del sistema lineare.
            assemble_system();
            solve_pde();

            // 2. Fase reattiva: aggiornamento dei gate ionici.
            solve_ode();
            
            // Sincronizza i ghost prima delle letture successive.
            locally_relevant_solution = solution;
            
            // Log periodico del progresso solo dal processo master.
            if (time_step_number % 100 == 0 && Utilities::MPI::this_mpi_process(mpi_communicator) == 0) {
                std::cout << "Step " << time_step_number << ", t=" << time << "ms" << std::endl;
            }
            
            // Barra di progresso a intervalli di 5%.
            if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0) {
                unsigned int progress = (time_step_number * 100) / total_steps;
                if (progress >= last_progress + 5) {
                    std::cout << "Progress: " << progress << "%" << std::endl;
                    last_progress = progress;
                }
            }
            
            // Calcolo del tempo di attivazione sui soli DoF owned.
            const double threshold = 0.5; // Soglia attivazione (modello adimensionale)
            
            for (const auto &cell : dof_handler.active_cell_iterators()) {
                if (cell->is_locally_owned()) {
                    std::vector<types::global_dof_index> local_dof_indices(fe.dofs_per_cell);
                    cell->get_dof_indices(local_dof_indices);

                    for (unsigned int i = 0; i < fe.dofs_per_cell; ++i) {
                        auto idx = local_dof_indices[i];
                        
                        // Evita scritture duplicate tra processi MPI.
                        if (locally_owned_dofs.is_element(idx)) {
                            // Registra il primo istante in cui il nodo supera la soglia.
                            if (activation_time(idx) == -1.0 && locally_relevant_solution(idx) > threshold) {
                                activation_time(idx) = time;
                            }
                        }
                    }
                }
            }
            activation_time.compress(VectorOperation::insert);

            // Esporta un risultato ogni 100 passi per limitare il numero di file.
            if (time_step_number % 100 == 0) {
                output_results(time_step_number);
            }
        }
        amg_preconditioner.reset(); // Libera la memoria dell'AMG alla fine della simulazione.

        // Messaggio finale emesso solo dal processo master.
        if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
            std::cout << "\rProgress: 100% - Simulation completed!" << std::endl;
    }
    
    // Output parallelo compatibile con ParaView: produce .vtu per processo e .pvtu master.
    template <int dim>
    void CardiacProblem<dim>::output_results(const unsigned int step) {
        DataOut<dim> data_out;
        data_out.attach_dof_handler(dof_handler);

        // Vettori esportati: potenziale e tempo di attivazione.
        data_out.add_data_vector(locally_relevant_solution, "V");
        data_out.add_data_vector(activation_time, "ActivationTime");
        data_out.build_patches();

        data_out.write_vtu_with_pvtu_record("./", "solution", step, mpi_communicator);
    }

    // Istanziazioni esplicite per i casi 2D e 3D supportati dal progetto.
    template class CardiacProblem<2>;
    template class CardiacProblem<3>;
}