#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
using namespace std;
#include "funcoes.h"

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