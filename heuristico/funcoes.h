#include <vector>
using namespace std;

struct Cidade
{
    int indice;
    float x;
    float y;
};

struct Tour
{
    float comprimento = 0;
    int quantidade; // quantidade de cidades
    int qualidade = 0;
    vector<Cidade> visitadas;
};

float distancia(Cidade a, Cidade b);
float comprimento(Tour t);
void returnOutput(Tour tour);
void leCidades(vector<Cidade> &cidades, int n);