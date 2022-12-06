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

//funcao para chiftar cidades em i posicoes
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

struct melhorTour {
    int quantidade;
    thrust::device_ptr<Cidade> ponteiro;

    //construtor
    melhorTour(thrust::device_ptr<Cidade> p, int q) : ponteiro(p), quantidade(q) {};

    __host__ __device__ double operator()(const int &id) const 
    {
        // pego o vetor com base no id
        thrust::device_ptr<Cidade> cidades_ptr = ponteiro + id * quantidade;

        // calculo a distancia total do vetor do ponteiro usando struct Cidade
        double distanciaTotal = 0.0;
        for (int i = 0; i < quantidade; i++) {
            distanciaTotal += Cidade()(cidades_ptr[i], cidades_ptr[(i + 1)]);
        }
        // adiciona distancia retorno
        distanciaTotal += Cidade()(cidades_ptr[quantidade - 1], cidades_ptr[0]);

        double melhorDistancia = distanciaTotal;

        // troco as cidades_ptr e vejo se vale a pena
        for (int i = 0; i < quantidade; i++) {
            thrust::swap(cidades_ptr[i], cidades_ptr[(i + 1)]);
            
            // calcula a distancia
            distanciaTotal = 0.0;
            for (int i = 0; i < quantidade; i++) {
                distanciaTotal += Cidade()(cidades_ptr[i], cidades_ptr[(i + 1)]);
            }
            // adiciona distancia retorno
            distanciaTotal += Cidade()(cidades_ptr[quantidade - 1], cidades_ptr[0]);

            double novaDistancia = distanciaTotal;
            if (novaDistancia < melhorDistancia) {
                melhorDistancia = novaDistancia;
            } else {
                thrust::swap(cidades_ptr[i], cidades_ptr[i + 1]);
            }
        }

        // retorna o vetor de cidades definido pelo ponteiro
        return melhorDistancia;

    }
};


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
    thrust::device_vector<Cidade> cidadesEmbaralhadas(10 * quantidade * quantidade); // vetor de cidades na GPU para armazenar as cidades embaralhadas
    
    thrust::default_random_engine eng(10); // seto o seed
    
    // solucao atual
    thrust::device_vector<Cidade> sequenciaAtual = dcidades;

    // gero 10N solucoes vizinhas e pego a melhor
    for (int i = 0; i < 10 * quantidade; i++) {

        // embaralho a solucao atual
        thrust::shuffle(dcidades.begin(), dcidades.end(), eng);

        // copio a solucao embaralhada para o vetor de cidades embaralhadas
        thrust::copy(dcidades.begin(), dcidades.end(), cidadesEmbaralhadas.begin() + (i * quantidade));
    }

    // pego o ponteiro para o vetor de cidades embaralhadas
    thrust::device_ptr<Cidade> cidadesEmbaralhadasPtr = &cidadesEmbaralhadas[0];

    // crio um vetor de indices para usar no transform
    thrust::device_vector<int> indices(10 * quantidade);
    thrust::sequence(indices.begin(), indices.end());

    // crio um vetor de distancias para usar no transform
    thrust::device_vector<double> melhoresDistancias(10 * quantidade);

    // crio um vetor de vizinhos
    thrust::device_vector<thrust::device_vector<Cidade>> melhoresVizinhos(10 * quantidade);

    // preencho o vetor de melhores vizinhos com os melhores vizinhos
    thrust::transform(indices.begin(), indices.end(), melhoresDistancias.begin(), melhorTour(cidadesEmbaralhadasPtr, quantidade));

    // printo todas as distancias
    for (int i = 0; i < 10 * quantidade; i++) {
        cout << "distancia: " << melhoresDistancias[i] << endl;
    }

    // pego o valor minimo do vetor de distancias usando minimum_element
    //thrust::device_vector<double>::iterator minDistancia = thrust::min_element(melhoresDistancias.begin(), melhoresDistancias.end());
    double melhorDistancia = thrust::reduce(melhoresDistancias.begin(), melhoresDistancias.end(), 1000000000000, thrust::minimum<double>());
    
    // printo a menor distancia
    cout << "menor distancia: " << melhorDistancia << endl;
    
    // // print output
    // retornaOutput(hmelhorVizinho, melhorDistancia);
}