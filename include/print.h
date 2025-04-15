void printPtrVet(double* vet, int maxvar, bool flag){
	cout << "{";
	for(int i = 0; i < maxvar; i++){
		cout << (i == 0 ? "" : " ") << vet[i];  // Acessa diretamente cada valor no vetor
		if (i < maxvar - 1) cout << ", ";  // Coloca uma vírgula entre os elementos
	}
	cout << "}" << (flag ? "M" : "E") << endl;
}


void printVet(vector<double*> vet, int maxvar, bool flag){
	cout << "{";
	for(int i=0; i<vet.size(); i++){
		cout << (i == 0 ? "" : " ") << "{";
		for(int j=0; j<g_problem_size; j++){
			printf("%5.1lf%s",vet[i][j], (j<g_problem_size-1 ? ", " : ""));
		}
		cout << "}" << (i<vet.size()-1? "\n":" ");
	}   cout << "}" << (flag?"M":"E") << endl;
}

void printVetMat(vector<double> vet, int col, bool flag){
	cout << "{";
	for(int i=0; i<(vet.size()/col); i++){
		cout << (i==0 ? "":" ") << "{";
		for(int j=0; j<col-1; j++){
			printf("%4.1lf%s", vet[i*col+j], (j<g_problem_size-1 ? ", " : " "));
		}

		printf("| %.1lf", vet[i*col + (col-1)]);
		cout << "}" << (i<(vet.size()/col)-1? "\n":"");
	}   cout << "}" << (flag?"M":"E") << endl;
}

void printPopMat(vector<pair<vector<double>, double>>& pop, int col, bool flag){
	cout << "\n\n\n\n{";

	for(int i=0; i<pop.size(); i++){
		cout << (i==0 ? "":" ") << "{";
		for(int j=0; j<pop[i].first.size(); j++){
			printf("%3.0lf%s", pop[i].first[j], (j<g_problem_size-1 ? ", " : " "));
		}

		printf("| %.1lf", pop[i].second);
		cout << "}" << ( i<pop.size() ? "\n":"");
	}   cout << "}" << (flag?"Mestre":"E") << endl;
}