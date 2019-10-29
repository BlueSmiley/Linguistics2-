#include<vector>
#include<string>
#include<iostream>
#include<iomanip>
#include<cmath>

using namespace std;

// want to represents vocab items by integers because then various tables 
// need by the IBM model and EM training can just be represented as 2-dim 
// tables indexed by integers

// the following #defines, defs of VS, VO, S, O, and create_vocab_and_data()
// are set up to deal with the specific case of the two pair corpus
// (la maison/the house)
// (la fleur/the flower)

// S VOCAB
#define LA 0
#define MAISON 1
#define FLEUR 2
// O VOCAB
#define THE 0
#define HOUSE 1
#define FLOWER 2

#define VS_SIZE 3
#define VO_SIZE 3
#define D_SIZE 2

#define NUM_ITERS 3


vector<string> VS(VS_SIZE); // S vocab: VS[x] gives Src word coded by x 
vector<string> VO(VO_SIZE); // O vocab: VO[x] gives Obs word coded by x

vector<vector<int> > S(D_SIZE); // all S sequences; in this case 2
vector<vector<int> > O(D_SIZE); // all O sequences; in this case 2

// sets S[0] and S[1] to be the int vecs representing the S sequences
// sets O[0] and O[1] to be the int vecs representing the O sequences
void create_vocab_and_data();

// functions which use VS and VO to 'decode' the int vecs representing the 
// Src and Obs sequences
void show_pair(int d);
void show_O(int d);
void show_S(int d);

int main() {
	create_vocab_and_data();

	// guts of it to go here
	// you may well though want to set up further global data structures
	// and functions which access them 

	// The probabilities each iteration
	double probs[VO_SIZE][VS_SIZE];
	// temporary storage for the counts of each word alignment pairs
	double counts[VO_SIZE][VS_SIZE];

	// initialise all probabilities to 1/3 because 3 seperate words and we don't know anything about data
	// Could be any value afaik
	for (int i = 0; i < VO_SIZE; i++) {
		for (int j = 0; j < VS_SIZE; j++) {
			probs[i][j] = 1.0 / 3.0;
		}
	}

	for (int iters = 0; iters < NUM_ITERS; iters++) {

		// reinitialise counts to 0
		for (int i = 0; i < VO_SIZE; i++) {
			for (int j = 0; j < VS_SIZE; j++) {
				counts[i][j] = 0;
			}
		}

		// Expectation
		// for each parralel corpus
		for (int p = 0; p < D_SIZE; p++) {
			std::vector<int> observed = O[p];
			std::vector<int> source = S[p];

			// Calculate total probability of observing that word in corpus = sum of P(O|S1) + P(O|S2)
			for (int o = 0; o < observed.size(); o++) {
				int obsWord = observed[o];
				double probObsWord = 0;

				//calculate actual probability of observing word Oj in sentence O
				for (int s = 0; s < source.size(); s++) {
					int srcWord = source[s];
					probObsWord += probs[obsWord][srcWord];
				}

				// Calculate p((oi,sj)|O,S)
				for (int s = 0; s < source.size(); s++) {
					int srcWord = source[s];
					if (probs[obsWord][srcWord] > 0) {
						counts[obsWord][srcWord] += probs[obsWord][srcWord] / probObsWord;
					}
					else {
						counts[obsWord][srcWord] += 0;
					}
				}

			}
		}


		// Maximisation
		for (int srcWord = 0; srcWord < VS_SIZE; srcWord++) {
			// this could be 0 if no words from source map to this word in observed
			double normalisationFactor = 0;
			for (int obsWord = 0; obsWord < VO_SIZE; obsWord++) {
				normalisationFactor += counts[obsWord][srcWord];
			}

			if (normalisationFactor <= 0) {
				for (int obsWord = 0; obsWord < VO_SIZE; obsWord++) {
					probs[obsWord][srcWord] = normalisationFactor;
				}
			}
			else {
				for (int obsWord = 0; obsWord < VO_SIZE; obsWord++) {
					probs[obsWord][srcWord] = counts[obsWord][srcWord] / normalisationFactor;
				}
			}
		}

		cout << "Counts:" << endl;
		for (int o = 0; o < VO_SIZE; o++) {
			for (int s = 0; s < VS_SIZE; s++) {
				cout << counts[o][s] << " ";
			}
			cout << endl;
		}
		cout << "Probabilities:" << endl;
		for (int o = 0; o < VO_SIZE; o++) {
			for (int s = 0; s < VS_SIZE; s++) {
				cout << probs[o][s] << " ";
			}
			cout << endl;
		}
		cout << endl;
	}

}

void create_vocab_and_data() {

	VS[LA] = "la";
	VS[MAISON] = "maison";
	VS[FLEUR] = "fleur";

	VO[THE] = "the";
	VO[HOUSE] = "house";
	VO[FLOWER] = "flower";

	cout << "source vocab\n";
	for (int vi = 0; vi < VS.size(); vi++) {
		cout << VS[vi] << " ";
	}
	cout << endl;
	cout << "observed vocab\n";
	for (int vj = 0; vj < VO.size(); vj++) {
		cout << VO[vj] << " ";
	}
	cout << endl;

	// make S[0] be {LA,MAISON}
	//      O[0] be {THE,HOUSE}
	S[0].resize(2);   O[0].resize(2);
	S[0] = { LA,MAISON };
	O[0] = { THE,HOUSE };

	// make S[1] be {LA,FLEUR}
	//      O[1] be {THE,FLOWER}
	S[1].resize(2);   O[1].resize(2);
	S[1] = { LA,FLEUR };
	O[1] = { THE,FLOWER };

	for (int d = 0; d < S.size(); d++) {
		show_pair(d);
	}
}

void show_O(int d) {
	for (int i = 0; i < O[d].size(); i++) {
		cout << VO[O[d][i]] << " ";
	}
}

void show_S(int d) {
	for (int i = 0; i < S[d].size(); i++) {
		cout << VS[S[d][i]] << " ";
	}
}

void show_pair(int d) {
	cout << "S" << d << ": ";
	show_S(d);
	cout << endl;
	cout << "O" << d << ": ";
	show_O(d);
	cout << endl;
}