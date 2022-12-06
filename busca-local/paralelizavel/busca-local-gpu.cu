%%writefile drive/MyDrive/supercompproj/busca-local.cu
// tsp - busca local em GPU

#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <thrust/shuffle.h>
using namespace std;

struct Cidade {
    double x, y;
    int id;

    __host__ __device__ double operator()(const Cidade &a, const Cidade &b) const {
        return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2));
    }
};

// funcao para chiftar cidades em i posicoes
thrust::device_vector<Cidade> shiftaCidades(thrust::device_vector<Cidade> cidades, int i) {
    thrust::device_vector<Cidade> cidadesShiftadas(cidades.size());
    thrust::copy(cidades.begin() + i, cidades.end(), cidadesShiftadas.begin());
    thrust::copy(cidades.begin(), cidades.begin() + i, cidadesShiftadas.begin() + (cidades.size() - i));
    
    return cidadesShiftadas;
}

double calculaDistanciaTotal(thrust::device_vector<Cidade> cidades) {
    // crio um vetor de cidades shiftadas
    thrust::device_vector<Cidade> cidadesShiftadas = shiftaCidades(cidades, 1);

    // calculo a distancia entre cada cidade e a proxima usando a funcao Cidade
    thrust::device_vector<double> distancias(cidades.size());
    thrust::transform(cidades.begin(), cidades.end(), cidadesShiftadas.begin(), distancias.begin(), Cidade());

    // somo todas as distancias usando reduce
    return thrust::reduce(distancias.begin(), distancias.end(), 0.0, thrust::plus<double>());
}

thrust::host_vector<Cidade> recebeInput(int quantidade)
{
    thrust::host_vector<Cidade> cidades(quantidade);
    for (int i = 0; i < quantidade; i++)
    {
        Cidade cidade;
        cin >> cidade.x >> cidade.y;
        cidade.id = i;
        cidades[i] = cidade;
    }

    return cidades;
}

void retornaOutput(thrust::host_vector<Cidade> cidades, double distancia)
{
    cout << distancia << " " << 0 << endl;
    for (int i = 0; i < cidades.size(); i++)
    {
        cout << cidades[i].id << " ";
    }
    cout << endl;
}


int main() {
    // read file
    int quantidade;
    cin >> quantidade;

    thrust::host_vector<Cidade> hcidades = recebeInput(quantidade); // vetor de cidades na CPU
    thrust::device_vector<Cidade> dcidades = hcidades; // vetor de cidades na GPU
    
    thrust::default_random_engine eng(10); // seto o seed

    // seto melhor sequencia de cidades e melhor distancia inicialmente
    thrust::device_vector<Cidade> melhorSequencia = dcidades;
    double melhorDistancia = calculaDistanciaTotal(melhorSequencia);
    
    // solucao atual
    thrust::device_vector<Cidade> sequenciaAtual = dcidades;

    // gero 10N solucoes vizinhas e pego a melhor
    for (int i = 0; i < 10 * quantidade; i++) {

        // embaralho a solucao atual
        thrust::shuffle(sequenciaAtual.begin(), sequenciaAtual.end(), eng);

        // calculo a distancia da solucao atual
        double distanciaAtual = calculaDistanciaTotal(sequenciaAtual);

        for (int l=0; l<quantidade; l++) {
            for (int r=0; r<quantidade; r++) {
                if (l!=r) {
                    // troco as duas cidades com swap
                    thrust::swap(sequenciaAtual[l], sequenciaAtual[r]);

                    // calculo a distancia da solucao atual
                    double novaDistancia = calculaDistanciaTotal(sequenciaAtual);

                    // se a nova distancia for menor que a atual, atualizo a melhor sequencia
                    if (novaDistancia < distanciaAtual) {
                        distanciaAtual = novaDistancia;
                    } else {
                        // se nao, desfaco a troca
                        thrust::swap(sequenciaAtual[l], sequenciaAtual[r]);
                    }
                }

            }
        }

        // se a distancia da solucao atual for menor que a melhor distancia, atualizo a melhor sequencia
        if (distanciaAtual < melhorDistancia) {
            melhorDistancia = distanciaAtual;
            melhorSequencia = sequenciaAtual;
        }
    }

    // retorno a melhor sequencia para a CPU
    thrust::host_vector<Cidade> hmelhorSequencia = melhorSequencia;

    // print output
    retornaOutput(hmelhorSequencia, melhorDistancia);
}