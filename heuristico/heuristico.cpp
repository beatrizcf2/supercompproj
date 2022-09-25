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
    // capacidade, peso atual, valor atual --> consigo infos a partir do vetor
    float comprimento = 0;
    int quantidade; // quantidade de cidades
    int qualidade = 0;
    vector<Cidade> visitadas;
};

float distancia(Cidade a, Cidade b)
{
    return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2));
}

int main()
{
    Tour tour;
    cin >> tour.quantidade; // numero de cidades a serem visitadas
    vector<Cidade> cidades;

    // Adiciono todas as cidades na lista de cidades possiveis
    for (int i = 0; i < tour.quantidade; i++)
    {
        Cidade cidade;
        cidade.indice = i;
        cin >> cidade.x >> cidade.y;
        cidades.push_back(cidade);
    }

    tour.visitadas.push_back(cidades[0]); // adiciona a cidade inicial ao tour
    cidades.erase(cidades.begin());       // remove a cidade inicial da lista de cidades possiveis


    float menorDistancia;
    int indiceMenorDistancia;
    // Calculo qual a cidade mais proxima da ultima cidade visitada
    for (int i = 0; i < tour.quantidade && cidades.size()>0; i++)
    {
        for (int j = 0; j < cidades.size(); j++)
        {
            if (j == 0)
            {
                menorDistancia = distancia(tour.visitadas[i], cidades[j]);
                indiceMenorDistancia = j;
            }
            else if (distancia(tour.visitadas[i], cidades[j]) < menorDistancia)
            {
                menorDistancia = distancia(tour.visitadas[i], cidades[j]);
                indiceMenorDistancia = j;
            }
        }
        tour.visitadas.push_back(cidades[indiceMenorDistancia]); // adiciona a cidade mais proxima ao tour
        tour.comprimento += menorDistancia;
        cidades.erase(cidades.begin() + indiceMenorDistancia); // remove a cidade da lista de cidades possiveis
    }

    //volta pra cidade inicial
    tour.comprimento += distancia(tour.visitadas[tour.quantidade-1], tour.visitadas[0]);
    

    // output
    cout << tour.comprimento << " " << 0 << endl;
    for (int i = 0; i < tour.visitadas.size(); i++)
    {
        cout << tour.visitadas[i].indice << " ";
    }
    cout << endl;

    return 0;
}