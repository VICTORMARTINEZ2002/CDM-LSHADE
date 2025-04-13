#include <mpi.h>

#include "de.h"

// Variaveis Funções de Benchmark
double *OShift,*M,*y,*z,*x_bound;
int ini_flag=0,n_flag,func_flag,*SS;

int g_function_number;
int g_problem_size;
unsigned int g_max_num_evaluations;

int    g_pop_size;
double g_arc_rate;
int    g_memory_size;
double g_p_best_rate;

int main(int argc, char **argv){
	int num_runs = 51; //number of runs
	g_problem_size = 10; //dimension size. please select from 10, 30, 50, 100
	g_max_num_evaluations = (g_problem_size*10000); //available number of fitness evaluations 

	srand((unsigned)time(NULL));
	cout << scientific << setprecision(8);

	// Inicialização MPI
	MPI_Status status;

	MPI_Init(NULL, NULL);
	int rank, size;
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	//L-SHADE parameters
	g_pop_size    = (int)round(18*g_problem_size);
	g_memory_size = 6;
	g_arc_rate    = 2.6;
	g_p_best_rate = 0.11;

	// DM-L-SHADE parameters
	double          elite_rate = 0.1;
	double       clusters_rate = 0.1468;
	int mining_generation_step = 168;

	for(int func=9; func<=9; func++){
		g_function_number = func;//+1;
		cout << "\n-------------------------------------------------------" << endl;
		cout << "Function = " << g_function_number << ", Dimension size = " << g_problem_size << "\n" << endl;

		double *bsf_fitness_array = (double*)malloc(sizeof(double) * num_runs);
		double mean_bsf_fitness   = 0;
		double std_bsf_fitness    = 0;

		for(int curr_run=0; curr_run<num_runs; curr_run++){ 
			// searchAlgorithm *alg = new LSHADE();
			int max_elite_size     = std::round(elite_rate * g_pop_size);
			int number_of_patterns = std::round(elite_rate * g_pop_size); // [TODO - revisar]
			DMLSHADE *alg          = new DMLSHADE(max_elite_size, number_of_patterns, mining_generation_step);

			// Inicio do Run
			alg->initializeParameters();
			alg->setSHADEParameters();
			int nfes = 0;

			// Cria os vetores de Fitness
			vector<double*> pop;
			vector<double> fitness(alg->pop_size, 0);
			vector<double*> children;
			vector<double> children_fitness(alg->pop_size, 0);

			// Inicializa População
			for(int i=0; i<alg->pop_size; i++){
				pop.push_back(alg->makeNewIndividual());
				children.push_back((double*)malloc(alg->problem_size*sizeof(double)));
			} alg->evaluatePopulation(pop, fitness); // Avalia Fitness População

			// BEST SOLUTION
			double* bsf_solution = (double*)malloc(alg->problem_size*sizeof(double));
			double bsf_fitness;

			// Encontra melhor solução na população [TODO - vai virar função]
			bsf_fitness = ((fitness[0]-alg->optimum)<alg->epsilon) ? alg->optimum : fitness[0];
			for(int i=0; i<alg->problem_size; i++){bsf_solution[i] = pop[0][i];}

			for(int i=0; i<alg->pop_size; i++){

				if((fitness[i]-alg->optimum) < alg->epsilon){fitness[i]=alg->optimum;}

				if(fitness[i] < bsf_fitness){
					bsf_fitness = fitness[i];
					for(int j=0; j < alg->problem_size; j++){bsf_solution[j]=pop[i][j];}
				}

				if((++nfes) >= alg->max_num_evaluations){break;}
			}
			////////////////////////////////////////////////////////////////////////////


			// for external archive
			int arc_ind_count = 0;  // Contador Individuos do Arquivo
			int random_selected_arc_ind;

			// Crio Arquivo
			vector<double*> archive;
			for(int i=0; i < alg->arc_size; i++){
				archive.push_back((double*)malloc(alg->problem_size*sizeof(double)));
			}

			int num_success_params;
			vector<double> success_sf;
			vector<double> success_cr;
			vector<double> dif_fitness;

			// the contents of M_f and M_cr are all initialiezed 0.5
			vector<double> memory_sf(alg->memory_size, 0.5);
			vector<double> memory_cr(alg->memory_size, 0.5);

			double temp_sum_sf;
			double temp_sum_cr;
			double sum;
			double weight;

			// memory index counter
			int memory_pos = 0;

			// for new parameters sampling
			double mu_sf, mu_cr;
			int random_selected_period;
			double *pop_sf = (double*)malloc(sizeof(double) * alg->pop_size);
			double *pop_cr = (double*)malloc(sizeof(double) * alg->pop_size);

			// for current-to-pbest/1
			int p_best_ind;
			int p_num = round(alg->pop_size * alg->p_best_rate);
			int *sorted_array =    (int*)malloc(   sizeof(int)*alg->pop_size);
			double *temp_fit  = (double*)malloc(sizeof(double)*alg->pop_size);

			// for linear population size reduction
			int max_pop_size = alg->pop_size;
			int min_pop_size = 4;
			int plan_pop_size;

			// Patterns set 
			vector<map<int, double>> patterns;

			// main loop
			while(nfes < alg->max_num_evaluations){
				alg->generation++;
				for(int i=0; i<alg->pop_size; i++){sorted_array[i]=i;}
				for(int i=0; i<alg->pop_size; i++){temp_fit[i]=fitness[i];}

				// Sorted: sorted_array ordenado com a posição dos melhores individuos;
				// [2,3,1,0] -> o indv da pos 2 é o melhor, 0 pior;
				alg->sortIndexWithQuickSort(&temp_fit[0], 0, alg->pop_size-1, sorted_array);

				// Mining steps
				alg->updateElite(pop, fitness, sorted_array);
				if(alg->generation % mining_generation_step == 0){
					patterns = alg->minePatterns(); // [TODO - Abrir pro mestre]

					// Inseir na populção os padrões
					for(size_t i=0; i<min((int)patterns.size(), alg->pop_size); i++){
						int idx = sorted_array[(alg->pop_size-1)-i];
						for(size_t j=0; j<alg->problem_size; j++){pop[idx][j] = patterns[i][j];}
					}
				}

				// A PARTIR DAQUI A GENTE ABSTRAI QUE ISSO É O LSHADE e somos felizes
				for(int target = 0; target < alg->pop_size; target++){
					// In each generation, CR_i and F_i used by each individual x_i are generated by first selecting 
					// an index r_i randomly from [1, H]
					random_selected_period = rand() % alg->memory_size;
					mu_sf = memory_sf[random_selected_period];
					mu_cr = memory_cr[random_selected_period];

					// generate CR_i and repair its value
					if(mu_cr == -1){
						pop_cr[target] = 0;
					}else{
						pop_cr[target] = alg->gauss(mu_cr, 0.1);
						if(pop_cr[target] > 1)
							pop_cr[target] = 1;
						else if(pop_cr[target] < 0)
							pop_cr[target] = 0;
					}

					// generate F_i and repair its value
					do
					{
						pop_sf[target] = alg->cauchy_g(mu_sf, 0.1);
					} while (pop_sf[target] <= 0);

					if(pop_sf[target] > 1)
						pop_sf[target] = 1;

					// p-best individual is randomly selected from the top alg->pop_size *  p_i members
					p_best_ind = sorted_array[rand() % p_num];
					alg->operateCurrentToPBest1BinWithArchive(pop, &children[target][0], target, p_best_ind, pop_sf[target], pop_cr[target], archive, arc_ind_count);
				}

				// evaluate the children's fitness values
				alg->evaluatePopulation(children, children_fitness);

				/////////////////////////////////////////////////////////////////////////
				// update the bsf-solution and check the current number of fitness evaluations
				//  if the current number of fitness evaluations over the max number of fitness evaluations, the search is terminated
				//  So, this program is unconcerned about L-SHADE algorithm directly
				for(int i=0; i<alg->pop_size; i++){
					nfes++;

					// following the rules of CEC 2014 real parameter competition,
					// if the gap between the error values of the best solution found and the optimal solution was 10^{−8} or smaller,
					// the error was treated as 0
					if((children_fitness[i] - alg->optimum) < alg->epsilon){children_fitness[i] = alg->optimum;}

					if(children_fitness[i] < bsf_fitness){
						bsf_fitness = children_fitness[i];
						for(int j=0; j < alg->problem_size; j++){bsf_solution[j] = children[i][j];}
					}

					// if(nfes % 1000 == 0){
					// //      cout << nfes << " " << bsf_fitness - alg->optimum << endl;
					// 	cout << bsf_fitness - alg->optimum << endl;
					// }
					if(nfes >= alg->max_num_evaluations)
						break;
				}
				////////////////////////////////////////////////////////////////////////////

				// generation alternation
				for(int i=0; i<alg->pop_size; i++){
					if(children_fitness[i] == fitness[i])
					{
						fitness[i] = children_fitness[i];
						for(int j=0; j < alg->problem_size; j++)
							pop[i][j] = children[i][j];
					}
					else if(children_fitness[i] < fitness[i])
					{
						// parent vectors x_i which were worse than the trial vectors u_i are preserved
						if(alg->arc_size > 1)
						{
							if(arc_ind_count < alg->arc_size)
							{
								for(int j=0; j < alg->problem_size; j++)
									archive[arc_ind_count][j] = pop[i][j];
								arc_ind_count++;
							}
							// Whenever the size of the archive exceeds, randomly selected elements are deleted to make space for the newly inserted elements
							else
							{
								random_selected_arc_ind = rand() % alg->arc_size;
								for(int j=0; j < alg->problem_size; j++)
									archive[random_selected_arc_ind][j] = pop[i][j];
							}
						}

						dif_fitness.push_back(fabs(fitness[i] - children_fitness[i]));
						fitness[i] = children_fitness[i];
						for(int j=0; j < alg->problem_size; j++)
							pop[i][j] = children[i][j];

						// successful parameters are preserved in S_F and S_CR
						success_sf.push_back(pop_sf[i]);
						success_cr.push_back(pop_cr[i]);
					}
				}

				num_success_params = success_sf.size();

				// if numeber of successful parameters > 0, historical memories are updated
				if(num_success_params > 0){
					memory_sf[memory_pos] = 0;
					memory_cr[memory_pos] = 0;
					temp_sum_sf = 0;
					temp_sum_cr = 0;
					sum = 0;

					for(int i=0; i < num_success_params; i++)
						sum += dif_fitness[i];

					// weighted lehmer mean
					for(int i=0; i < num_success_params; i++){
						weight = dif_fitness[i] / sum;

						memory_sf[memory_pos] += weight * success_sf[i] * success_sf[i];
						temp_sum_sf += weight * success_sf[i];

						memory_cr[memory_pos] += weight * success_cr[i] * success_cr[i];
						temp_sum_cr += weight * success_cr[i];
					}

					memory_sf[memory_pos] /= temp_sum_sf;

					if(temp_sum_cr == 0 || memory_cr[memory_pos] == -1){
						memory_cr[memory_pos] = -1;
					}else{
						memory_cr[memory_pos] /= temp_sum_cr;
					}

					// increment the counter
					memory_pos++;
					if(memory_pos >= alg->memory_size){memory_pos = 0;}

					// clear out the S_F, S_CR and delta fitness
					success_sf.clear();
					success_cr.clear();
					dif_fitness.clear();
				}

				// calculate the population size in the next generation
				plan_pop_size = round((((min_pop_size - max_pop_size) / (double)alg->max_num_evaluations) * nfes) + max_pop_size);

				if(alg->pop_size > plan_pop_size){
					alg->reduction_ind_num = alg->pop_size - plan_pop_size;
					if(alg->pop_size - alg->reduction_ind_num < min_pop_size)
						alg->reduction_ind_num = alg->pop_size - min_pop_size;

					alg->reducePopulationWithSort(pop, fitness);

					// resize the archive size
					alg->arc_size = alg->pop_size * g_arc_rate;
					if(arc_ind_count > alg->arc_size)
						arc_ind_count = alg->arc_size;

					// resize the number of p-best individuals
					p_num = round(alg->pop_size * alg->p_best_rate);
					if(p_num <= 1)
						p_num = 2;
				}
			}

			free(temp_fit);

			// return bsf_fitness - alg->optimum;

			bsf_fitness_array[curr_run] = bsf_fitness - alg->optimum;
			cout << curr_run + 1 << "th run, " << "error value = " << bsf_fitness_array[curr_run] << endl;
		}

		// Impressão dos Resultados
		for(int j=0; j<num_runs; j++){mean_bsf_fitness += bsf_fitness_array[j];}
		mean_bsf_fitness /= num_runs;

		for(int j=0; j<num_runs; j++){std_bsf_fitness += pow((mean_bsf_fitness - bsf_fitness_array[j]), 2.0);}
		std_bsf_fitness /= num_runs;
		std_bsf_fitness = sqrt(std_bsf_fitness);

		cout  << "\nmean = " << mean_bsf_fitness << ", std = " << std_bsf_fitness << endl;
		free(bsf_fitness_array);
	}


	MPI_Finalize();
	return 0;
}
