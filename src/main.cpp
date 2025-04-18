#include <mpi.h>
#include <limits>
#include <cstdlib>   // Clear System
#include <algorithm> // Sort

#include <pyclustering/cluster/kmeans.hpp>
#include <pyclustering/cluster/xmeans.hpp>
#include <pyclustering/cluster/random_center_initializer.hpp>

using namespace pyclustering;
using namespace pyclustering::clst;

#include "de.h"
#include "print.h"
#include "statistics.h"

#define RANK_MESTRE 0
#define RANK_LSHADE 1

#define TAG_FINALZ 0
#define TAG_MESTRE 1
#define TAG_LSHADE 2

// Variaveis Funções de Benchmark
double *OShift,*M,*y,*z,*x_bound;
int ini_flag=0,n_flag,func_flag,*SS;

int g_function_number;
int g_problem_size;
int g_fator_max_evaluations;
int g_fator_pop_size;
double g_fator_slaveSize;
unsigned int g_max_num_evaluations;

int    g_pop_size;
double g_arc_rate;
int    g_memory_size;
double g_p_best_rate;

// Diversidade
int g_flag_diversidade;


// Função para Inserção de Individuos na população do Mestre
// Para considerar diversidade, passe o número de processos size
// Para uma abordagem Meritocratica ("Elite das Elites"), passe 1 como size
void inserirPopMestre(vector<pair<vector<double>, double>>& pop, vector<double>& elite, int slaveSize, int rank, int size){
	int popSize = pop.size();
	int tam = elite.size()/(g_problem_size+1);

	size = max(size,2);
	vector<int> intv(size); //3 Escravos -> [0,3,6,9]
	for(int i=0; i<intv.size(); i++){intv[i]=i*slaveSize;}

	for(int i=0; i<tam; i++){
		auto temp_init = elite.begin()+ i*(g_problem_size+1);
		vector<double> tempInd(temp_init, temp_init+g_problem_size);
		double temp_fitness = elite[(i+1)*(g_problem_size+1)-1];

		pop.insert(pop.begin(), {tempInd, temp_fitness}); // + intv[rank] + i // Insere no fim do intervalo
	}

	// Ordenação 
	if(size>2){ // Parcial
		sort(pop.begin()+intv[rank-1], pop.begin()+intv[rank]+tam, [](auto& a, auto& b){return a.second < b.second;});
		pop.erase(pop.begin()+intv[rank-1]+slaveSize, pop.begin()+intv[rank]+tam);		
	}else{ // Total
		sort(pop.begin(), pop.end(), [](auto& a, auto& b){return a.second < b.second;});
		pop.erase(pop.begin()+popSize, pop.end());
	}
	// TODO - COntrole tamanho da Pop
}

bool isTrue(vector<bool> vet){
	bool flag = true;
	for(int i=0; i<vet.size(); i++){flag &= vet[i];}
	return flag;
}


int main(int argc, char **argv){
	srand((unsigned)time(NULL));

//INICIALIZAÇÃO DE PARAMETROS
	// Leitura Parâmetros
	g_function_number       = std::atoi(argv[1]);
	g_problem_size          = std::atoi(argv[2]); // 10, 30, 50, 100
	g_fator_pop_size        = std::atoi(argv[3]);
	g_fator_slaveSize       = std::atof(argv[4]);
	g_fator_max_evaluations = std::atoi(argv[5]);
	g_flag_diversidade      = std::atoi(argv[6]);
	g_max_num_evaluations   = (g_fator_max_evaluations*g_problem_size); //available number of fitness evaluations 

	// Inicialização MPI
	int rank, size;
	int flagMensagem;
	MPI_Status status;
	MPI_Init(NULL, NULL);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	// BEST SOLUTION
	vector<double> bsf_solution(g_problem_size+1); // Inclui Fitness p/ Facilizar Envio P/ Mestre
	double bsf_fitness;

	//L-SHADE parameters
	g_pop_size    = (int)round(g_fator_pop_size*g_problem_size);
	g_memory_size = 6;
	g_arc_rate    = 2.6;
	g_p_best_rate = 0.11;

	// DM-L-SHADE parameters
	int number_of_patterns;
	vector<double> patterns; // Also MPI Parameter
	double          elite_rate = 0.1;
	double       clusters_rate = 0.1468;
	int mining_generation_step = 50; //168;


	// MPI Parametros
	vector<double> tempElite(std::round(elite_rate*g_pop_size)*(g_problem_size+1));
	int slaveSize  = max(3, static_cast<int>(round(g_pop_size * elite_rate * g_fator_slaveSize)));


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//													DIVISÃO RANKS													//
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	if(rank==0){ // Mestre
		// Impressão Parâmetros
		std::system("clear");
		cout << "//------------------------- DM-LSHADE PARALLEL (N="<<size<<") -------------------------//" << endl;
		cout << "Function        = " <<       g_function_number << endl;
		cout << "Dimension size  = " <<          g_problem_size << endl;
		cout << "Fator População = " <<        g_fator_pop_size << endl;
		cout << "Fator Avaliação = " << g_fator_max_evaluations << endl;
		cout << setprecision(8);

		vector<double> mineTime;
		double start = MPI_Wtime();
		vector<vector<double>> bestSolutions(max(1,size-1), vector<double>(g_problem_size+1));


		if(size==1){
			// [TODO] Mine Time
			DMLSHADE *alg = new DMLSHADE(std::round(elite_rate*g_pop_size), number_of_patterns, mining_generation_step);
			bestSolutions[0][g_problem_size] = alg->run();
			
		}else{
			int contFinlz=0;
			int newMine = false;
			int maxPop = (size-1)*elite_rate*g_pop_size;
			vector<pair<vector<double>, double>> pop(slaveSize*(size-1), {vector<double>(g_problem_size, 0.0), std::numeric_limits<double>::max()});

			vector<bool> slaveRecv(size-1, false);

			while(contFinlz<(size-1)){
				MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flagMensagem, &status);
				while(flagMensagem){
					if(status.MPI_TAG==TAG_LSHADE){
						int recvSize;
						MPI_Get_count(&status, MPI_DOUBLE, &recvSize);
						MPI_Recv(&tempElite[0], recvSize, MPI_DOUBLE, MPI_ANY_SOURCE, TAG_LSHADE, MPI_COMM_WORLD, &status);
						
						slaveRecv[status.MPI_SOURCE-1]=true;

						int tam = tempElite.size()/(g_problem_size+1);
						if(tam>=1){
							inserirPopMestre(pop, tempElite, slaveSize, status.MPI_SOURCE, (g_flag_diversidade?size:2));
							printPopMat(pop, g_problem_size, true);
						}
						if(isTrue(slaveRecv)){newMine = true;} // Ativa mineração [TODO - Criterio MIn de Mineração]

					}else if(status.MPI_TAG==TAG_FINALZ){
						MPI_Recv(&bsf_solution[0], (g_problem_size+1), MPI_DOUBLE, MPI_ANY_SOURCE, TAG_FINALZ, MPI_COMM_WORLD, &status);
						contFinlz++;
						bestSolutions[status.MPI_SOURCE-1] = bsf_solution;
						bestSolutions[status.MPI_SOURCE-1][g_problem_size] -= (100*g_function_number); // Fix Fitness;
					}

					MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flagMensagem, &status); 
				}

				if(newMine){
					printf("Nova Mineração\n");
					newMine = false;
					double startMine = MPI_Wtime();
					number_of_patterns = clusters_rate*(elite_rate*pop.size());
					vector<vector<double>> data;
					for(size_t i=0; i<pop.size(); i++){
						data.push_back(vector<double>(get<0>(pop[i]).begin(), get<0>(pop[i]).begin() + g_problem_size));
					}

					vector<vector<double> > start_centers;
					pyclustering::clst::xmeans_data output_result;
					long seed = 1;

					if(number_of_patterns>0){
						output_result.clusters().clear();
						random_center_initializer(number_of_patterns, seed).initialize(data, start_centers);
						kmeans solver(start_centers);
						solver.process(data, output_result);
					}else{
						random_center_initializer(1, seed).initialize(data, start_centers);
						xmeans solver(start_centers, pop.size(), 1e-4, splitting_type::BAYESIAN_INFORMATION_CRITERION, 1, seed);
						solver.process(data, output_result);
					}
					
					patterns.clear();
					auto centers = output_result.centers();
					for(auto center:centers){
						for(size_t j=0; j<g_problem_size; j++){
							patterns.push_back(center[j]);
						}
					}

					// Enviar em "Broadcast" os padroes para os Escravos
					if((patterns.size()/g_problem_size)>0){
						for(int i=1; i<size; i++){
							MPI_Send(&patterns[0], patterns.size(), MPI_DOUBLE, i, TAG_MESTRE, MPI_COMM_WORLD); 
						}
					}

					mineTime.push_back(MPI_Wtime()-startMine);
				}

			}
			
			//printPopMat(pop, g_problem_size, true);
			
		}

		// Verifica Melhor Solução
		int posBest=0;
		for(int i=1; i<bestSolutions.size(); i++){
			if(bestSolutions[i][g_problem_size] < bestSolutions[posBest][g_problem_size]){posBest=i;}
		}
		bsf_fitness  = bestSolutions[posBest][g_problem_size];
		//bsf_solution.assign(bestSolutions[posBest].begin(), bestSolutions[posBest].end()-1);
		
		double execTime = MPI_Wtime()-start;
		std::ostringstream pathfile;
		pathfile << "./logs/" << g_function_number << "-" << g_problem_size << "-" << size << ".log";
		std::ofstream file(pathfile.str(), ios::app); // Escrever ao Final 
		if(file.is_open()){
			file << execTime << endl;
			cout << "Log atualizado com sucesso!" << endl;
			file.close();
		}

		cout << "Melhor Fitness: "                      << bsf_fitness     << endl;
		cout << "Tempo Execução: "                      << execTime        << endl;
		cout << "Quantidade de Minerações: "            << mineTime.size() << endl;
		cout << "Média do Tempo de Mineração: "         << mean(mineTime)  << endl;
		cout << "Desvio Padrão do Tempo de Mineração: " << stdev(mineTime) << endl;


	}

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//													DIVISÃO RANKS													//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	if(rank>=1){
		DMLSHADE *alg = new DMLSHADE(std::round(elite_rate*g_pop_size), number_of_patterns, mining_generation_step);
		// searchAlgorithm *alg = new LSHADE();

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




	// LSHADE PARAMETROS
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

	// MAIN LOOP
		while(nfes < alg->max_num_evaluations){
			alg->generation++;
			for(int i=0; i<alg->pop_size; i++){sorted_array[i]=i;}
			for(int i=0; i<alg->pop_size; i++){temp_fit[i]=fitness[i];}

			// Sorted: sorted_array ordenado com a posição dos melhores individuos;
			alg->sortIndexWithQuickSort(&temp_fit[0], 0, alg->pop_size-1, sorted_array);

			// Mining steps
			alg->updateElite(pop, fitness, sorted_array);
			
			tempElite.clear();
			for(int i=0; i<alg->elite.size(); i++){
				for(int j=0; j<g_problem_size; j++){
					tempElite.push_back(get<0>(alg->elite[i])[j]);
				}	tempElite.push_back(get<1>(alg->elite[i])-alg->optimum);
			} 

			// printVetMat(tempElite, g_problem_size+1, false); // Individuo + Fitness

			// MPI ENVIA ELITE
			if(alg->generation % mining_generation_step == 0){
				// MPI ENVIA
				if(alg->elite.size()>0){
					MPI_Send(&tempElite[0], alg->elite.size()*(g_problem_size+1), MPI_DOUBLE, RANK_MESTRE, TAG_LSHADE, MPI_COMM_WORLD); 
				}

				// Inserir na população os padrões
				MPI_Iprobe(RANK_MESTRE, MPI_ANY_TAG, MPI_COMM_WORLD, &flagMensagem, &status);
				if(flagMensagem){
					int recvSize;
					do{ // Verificar se estou com a mensagem mais recente
						MPI_Get_count(&status, MPI_DOUBLE, &recvSize);
						patterns.resize(recvSize);
						MPI_Recv(&patterns[0], recvSize, MPI_DOUBLE, RANK_MESTRE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
						MPI_Iprobe(RANK_MESTRE, MPI_ANY_TAG, MPI_COMM_WORLD, &flagMensagem, &status);
					}while(flagMensagem);
					
					for(size_t i=0; i<min((int)patterns.size()/g_problem_size, alg->pop_size); i++){
						int idx = sorted_array[(alg->pop_size-1)-i];
						for(size_t j=0; j<alg->problem_size; j++){
							pop[idx][j] = patterns[i*g_problem_size+j];
						}
					}	

				}

			}				

		// RESTO DO LSHADE
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
				do{
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

				if((children_fitness[i] - alg->optimum) < alg->epsilon){children_fitness[i] = alg->optimum;}

				if(children_fitness[i] < bsf_fitness){
					bsf_fitness = children_fitness[i];
					for(int j=0; j < alg->problem_size; j++){bsf_solution[j] = children[i][j];}
				}

				if(nfes >= alg->max_num_evaluations){break;}
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
			if(num_success_params>0){
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
				if(p_num <= 1){p_num = 2;}
			}
		
		} // FIM MAIN WHILE

		
		// cout << "Melhor Solução = "; printPtrVet(bsf_solution, g_problem_size);

		free(temp_fit);
		// MPI Finalização
		bsf_solution[bsf_solution.size()-1] = bsf_fitness;
		MPI_Send(&bsf_solution[0], (g_problem_size+1), MPI_DOUBLE, RANK_MESTRE, TAG_FINALZ, MPI_COMM_WORLD);

	}

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//													DIVISÃO RANKS													//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


	// Impressão dos Resultados
	// for(int j=0; j<num_runs; j++){mean_bsf_fitness += bsf_fitness_array[j];}
	// mean_bsf_fitness /= num_runs;

	// for(int j=0; j<num_runs; j++){std_bsf_fitness += pow((mean_bsf_fitness - bsf_fitness_array[j]), 2.0);}
	// std_bsf_fitness /= num_runs;
	// std_bsf_fitness = sqrt(std_bsf_fitness);

	// cout  << "\nmean = " << mean_bsf_fitness << ", std = " << std_bsf_fitness << endl;
	// free(bsf_fitness_array);


	MPI_Finalize();
	return 0;
}
