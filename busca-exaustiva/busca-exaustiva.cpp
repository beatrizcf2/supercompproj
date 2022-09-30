#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
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

float distancia(Cidade a, Cidade b)
{
    return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2));
}

float comprimento(Tour t)
{
    float comprimento = 0;
    for (int i = 0; i < t.quantidade - 1; i++)
    {
        comprimento += distancia(t.visitadas[i], t.visitadas[i + 1]);
    }
    // voltar para a cidade inicial
    comprimento += distancia(t.visitadas[t.quantidade - 1], t.visitadas[0]);
    return comprimento;
}

void returnOutput(Tour tour){
    // output
    cout << tour.comprimento << " " << 0 << endl;
    for (int i = 0; i < tour.visitadas.size(); i++)
    {
        cout << tour.visitadas[i].indice << " ";
    }
    cout << endl;
}


void leCidades(vector<Cidade> &cidades, int n){
    for (int i = 0; i < n; i++)
    {
        Cidade cidade;
        cidade.indice = i;
        cin >> cidade.x >> cidade.y;
        cidades.push_back(cidade);
    }
}

void buscaExaustiva(vector<Cidade> cidades, Tour tour, Tour &melhorTour, int &n){
    if(cidades.size() == 0){
        n +=1 ;
        cerr << "num_leaf " << n << endl; 
        tour.comprimento = comprimento(tour);
        //returnOutput(tour);
        if (tour.comprimento < melhorTour.comprimento)
        {
            melhorTour = tour;
        }
        return;
    }

    for (int i = 0; i < cidades.size(); i++)
    {
        tour.visitadas.push_back(cidades[i]);
        cidades.erase(cidades.begin() + i);
        
        buscaExaustiva(cidades, tour, melhorTour, n);
        cidades.insert(cidades.begin() + i, tour.visitadas[tour.visitadas.size() - 1]);
        tour.visitadas.pop_back();
    }
}


int main(){
    Tour tour;
    cin >> tour.quantidade; // numero de cidades a serem visitadas
    vector<Cidade> cidades; // lista de todas as cidades

    // Adiciono todas as cidades na lista de cidades possiveis
    leCidades(cidades, tour.quantidade);

    //assumo que o melhor tour é o primeiro, a principio
    Tour melhorTour;
    melhorTour.quantidade = tour.quantidade;
    melhorTour.visitadas = cidades;
    melhorTour.comprimento = comprimento(melhorTour);

    int n = 0; // numero de solucoes possiveis

    // Faço a busca exaustiva
    buscaExaustiva(cidades, tour, melhorTour, n);

    // Retorno output do melhor tour
    returnOutput(melhorTour);

    return 0;
}
