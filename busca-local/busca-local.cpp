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

//uma função que avalie a qualidade (fitness) dessa solução

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

int main()
{
    //vector<Tour> solucoes;
    Tour tour;
    cin >> tour.quantidade; // numero de cidades a serem visitadas
    vector<Cidade> cidades; // lista de todas as cidades

    // Adiciono todas as cidades na lista de cidades possiveis
    for (int i = 0; i < tour.quantidade; i++)
    {
        Cidade cidade;
        cidade.indice = i;
        cin >> cidade.x >> cidade.y;
        cidades.push_back(cidade);
    }

    // Seto o seed como 10
    srand(10);

    // Gero uma solucao aleatoria
    for (int i = 0; i < tour.quantidade; i++)
    {
        // Sorteio um numero aleatorio entre 0 e o tamanho da lista de cidades
        int indice = rand() % cidades.size();
        // Adiciono a cidade sorteada ao tour
        tour.visitadas.push_back(cidades[indice]);
        // Removo a cidade sorteada da lista de cidades
        cidades.erase(cidades.begin() + indice);
        // Adiciono ao comprimento
        tour.comprimento += distancia(tour.visitadas[i], tour.visitadas[i+1]);
    }
    //solucoes.push_back(tour); // adiciono esse tour a minha lista de solucoes
    // cout << "Comprimento solucao aleatoria inicial: " << tour.comprimento << endl;

    // Gero 10N solucoes vizinhas
    // se for possível inverter a ordem de visitação de duas cidades e isso melhorar a solução então faça a troca
    // se for possivel trocar a ordem e melhorar a solução, faça a troca
    for (int i = 0; i < 10 * tour.quantidade; i++)
    {
        for (int l=0; l<tour.quantidade; l++){
            for (int r=0; r<tour.quantidade; r++){
                if (l != r){
                    // Inverte a ordem de visitacao de duas cidades
                    swap(tour.visitadas[l], tour.visitadas[r]);
                    float comp = comprimento(tour);
                    // Verifico se o comprimento é menor que o comprimento do tour atual
                    if (comp < tour.comprimento){
                        // Se for, atualizo o comprimento
                        // cout << "Melhorou o comprimento de " << tour.comprimento << " para " << comp << endl;
                        tour.comprimento = comp;
                    } else {
                        // Se não for, desfaco a troca
                        swap(tour.visitadas[l], tour.visitadas[r]);
                    }
                }
            }
        }
        // Adicionar ao comprimento a volta para a cidade origem
        tour.comprimento += distancia(tour.visitadas[tour.quantidade-1], tour.visitadas[0]);

        cerr << "local: " << tour.comprimento << " ";
        for (int i = 0; i < tour.quantidade; i++){
            cerr << tour.visitadas[i].indice << " ";
        }
        cerr << endl;
        //solucoes.push_back(tour);
    }

    // output
    returnOutput(tour);

    



    

    
    
    




    return 0;
}