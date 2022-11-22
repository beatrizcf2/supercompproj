// busca local paralelizada
// importa bibliotecas
#include <iostream>
#include <omp.h>
#include <vector>
#include <algorithm>
#include <cmath>
using namespace std;

// define struct cidade
struct Cidade
{
    int indice;
    float x;
    float y;
};

// define struct tour
struct Tour
{
    // capacidade, peso atual, valor atual --> consigo infos a partir do vetor
    float comprimento = 0;
    int quantidade; // quantidade de cidades
    int qualidade = 0;
    vector<Cidade> visitadas;
};

// funcao que calcula a distancia entre duas cidades
float distancia(Cidade a, Cidade b)
{
    return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2));
}

// funcao que calcula o comprimento de um tour
float comprimento(vector<Cidade> cidades, int quantidade)
{
    float comprimento = 0;
    for (int i = 0; i < quantidade - 1; i++)
    {
        comprimento += distancia(cidades[i], cidades[i + 1]);
    }
    // voltar para a cidade inicial
    comprimento += distancia(cidades[quantidade - 1], cidades[0]);
    return comprimento;
}

// funcao que retorna o output
void returnOutput(Tour tour)
{
    // output
    cout << tour.comprimento << " " << 0 << endl;
    for (int i = 0; i < tour.visitadas.size(); i++)
    {
        cout << tour.visitadas[i].indice << " ";
    }
    cout << endl;
}

// funcao que retorna uma solucao aleatoria
void retornaAleatoria(vector<Cidade> cidades, Tour &tour)
{
    // retorna uma solução aleatória
    random_shuffle(cidades.begin(), cidades.end());
    tour.visitadas = cidades;
    tour.comprimento = comprimento(tour.visitadas, tour.quantidade);
}

int main(){
    //vector<Tour> solucoes;
    Tour t, melhorTour;
    cin >> t.quantidade; // numero de cidades a serem visitadas
    vector<Cidade> cidades; // lista de todas as cidades

    // leitura das cidades
    for (int i = 0; i < t.quantidade; i++)
    {
        Cidade cidade;
        cidade.indice = i;
        cin >> cidade.x >> cidade.y;
        cidades.push_back(cidade);
    }

    float melhorComprimento = comprimento(cidades, t.quantidade);

    // Gero 10N solucoes vizinhas
    // se for possível inverter a ordem de visitação de duas cidades e isso melhorar a solução então faça a troca
    // se for possivel trocar a ordem e melhorar a solução, faça a troca
    // paralelizando a geração das soluções
    #pragma omp parallel
    {
    #pragma omp parallel for
        for (int i = 0; i < 10 * t.quantidade; i++)
        {
            Tour tour;
            tour.quantidade = t.quantidade;
            retornaAleatoria(cidades, tour);
            for (int l=0; l<tour.quantidade; l++){
                for (int r=0; r<tour.quantidade; r++){
                    if (l != r){
                        // Inverte a ordem de visitacao de duas cidades
                        swap(tour.visitadas[l], tour.visitadas[r]);
                        float comp = comprimento(tour.visitadas, tour.quantidade);
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
            if (tour.comprimento < melhorComprimento){
                #pragma omp critical
                {
                    if (tour.comprimento < melhorComprimento){
                        melhorComprimento = tour.comprimento;
                        melhorTour = tour;
                    }
                }
            }

            cerr << "local: " << tour.comprimento << " ";
            for (int i = 0; i < tour.quantidade; i++){
                cerr << tour.visitadas[i].indice << " ";
            }
            cerr << endl;
        }
    }
    

    // output
    returnOutput(melhorTour);
    return 0;
}
