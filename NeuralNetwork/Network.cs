using System;
using System.Linq;
using MathNet.Numerics;

namespace NeuralNetwork
{
    public class Network
    {
        private static readonly int _learningRate = 2;
        private static readonly Random _random = new Random(1234);
        private static readonly Func<double, double> _actFunc = new Func<double, double>(x => SpecialFunctions.Logistic(x));
        private static readonly Func<double, double> _actFuncDeriv = Differentiate.DerivativeFunc(_actFunc, 1);
        public Layer[] Layers { get; set; }

        public Network(int[] neuronCountPerLayer)
        {
            Layers = new Layer[neuronCountPerLayer.Length];
            for (int i = 0; i < Layers.Length; i++)
            {
                Layers[i] = new Layer(neuronCountPerLayer[i]);
            }

            for (int i = 1; i < Layers.Length; i++)
            {
                for (int j = 0; j < Layers[i].Neurons.Length; j++)
                {
                    Layers[i].Neurons[j].Weights = new double[Layers[i - 1].Neurons.Length];
                    Layers[i].Neurons[j].DCDW = new double[Layers[i].Neurons[j].Weights.Length];
                    for (int k = 0; k < Layers[i].Neurons[j].Weights.Length; k++)
                    {
                        Layers[i].Neurons[j].Weights[k] = 0; // 2 * _random.NextDouble() - 1;
                    }
                    Layers[i].Neurons[j].Bias = 0; // 2 * _random.NextDouble() - 1;
                }
            }
        }

        public double[] Guess(double[] input)
        {
            return Forward(input);
        }

        public void TrainExamples((double[], double[])[] examples)
        {
            for (int i = 0; i < examples.Length; i++)
            {
                TrainExample(examples[i].Item1, examples[i].Item2);
            }

            //var avgWeightDiffs = TotalWeightDiffs.Aggregate((sum, weight) => sum += weight) / TotalWeightDiffs.Count;
        }

        //FIXME: make private
        public double[] Forward(double[] input)
        {
            // Set input into activations of first layer
            for (int j = 0; j < Layers[0].Neurons.Length; j++)
            {
                Layers[0].Neurons[j].Activation = input[j];
            }

            for (int i = 1; i < Layers.Length; i++)
            {
                for (int j = 0; j < Layers[i].Neurons.Length; j++)
                {
                    double sum = 0;
                    for (int k = 0; k < Layers[i - 1].Neurons.Length; k++)
                    {
                        sum += Layers[i].Neurons[j].Weights[k] * Layers[i - 1].Neurons[k].Activation;
                    }
                    sum += Layers[i].Neurons[j].Bias;

                    Layers[i].Neurons[j].Activation = SpecialFunctions.Logistic(sum);
                }
            }

            return Layers[Layers.Length - 1].Neurons.Select(x => x.Activation).ToArray();
        }

        //FIXME: make private
        public double Cost(double[] output, double[] expectedOutput)
        {
            double cost = 0;
            for (int i = 0; i < output.Length; i++)
            {
                cost += Math.Pow(output[i] - expectedOutput[i], 2);
            }
            cost /= expectedOutput.Length;
            return cost;
        }

        //FIXME: make private
        public void BackPropagation(double[] expectedOutput)
        {
            ResetDerivatives();

            int L = Layers.Length - 1;

            var dC_da = 2 * (Layers[L].Neurons[0].Activation - expectedOutput[0]);

            var z = Layers[L - 1].Neurons[0].Activation * Layers[L].Neurons[0].Weights[0] + Layers[L].Neurons[0].Bias;
            var da_dz = _actFuncDeriv(z);
            var dz_dw = Layers[L - 1].Neurons[0].Activation;
            var dC_dw = dz_dw * da_dz * dC_da;
            Layers[L].Neurons[0].DCDW[0] += dC_dw;

            var dC_db = da_dz * dC_da;
            Layers[L].Neurons[0].DCDB += dC_db;


            var dz1_dw1 = Layers[L - 2].Neurons[0].Activation;
            var z1 = Layers[L - 2].Neurons[0].Activation * Layers[L - 1].Neurons[0].Weights[0] + Layers[L - 1].Neurons[0].Bias;
            var da1_dz1 = _actFuncDeriv(z1);
            var dz_da1 = Layers[L].Neurons[0].Weights[0];
            var dC_dw1 = dz1_dw1 * da1_dz1 * dz_da1 * da_dz * dC_da;
            Layers[L - 1].Neurons[0].DCDW[0] += dC_dw1;

            var dC_db1 = da1_dz1 * dz_da1 * da_dz * dC_da;
            Layers[L - 1].Neurons[0].DCDB += dC_db1;


            var dz2_dw2 = Layers[L - 3].Neurons[0].Activation;
            var z2 = Layers[L - 3].Neurons[0].Activation * Layers[L - 2].Neurons[0].Weights[0] + Layers[L - 2].Neurons[0].Bias;
            var da2_dz2 = _actFuncDeriv(z2);
            var dz1_da2 = Layers[L - 1].Neurons[0].Weights[0];
            var dC_dw2 = dz2_dw2 * da2_dz2 * dz1_da2 * da1_dz1 * dz_da1 * da_dz * dC_da;
            Layers[L - 2].Neurons[0].DCDW[0] += dC_dw2;

            var dC_db2 = da2_dz2 * dz1_da2 * da1_dz1 * dz_da1 * da_dz * dC_da;
            Layers[L - 2].Neurons[0].DCDB += dC_db2;


            Layers[L].Neurons[0].Weights[0] -= _learningRate * Layers[L].Neurons[0].DCDW[0];
            Layers[L].Neurons[0].Bias -= _learningRate * Layers[L].Neurons[0].DCDB;

            Layers[L - 1].Neurons[0].Weights[0] -= _learningRate * Layers[L - 1].Neurons[0].DCDW[0];
            Layers[L - 1].Neurons[0].Bias -= _learningRate * Layers[L - 1].Neurons[0].DCDB;

            Layers[L - 2].Neurons[0].Weights[0] -= _learningRate * Layers[L - 2].Neurons[0].DCDW[0];
            Layers[L - 2].Neurons[0].Bias -= _learningRate * Layers[L - 2].Neurons[0].DCDB;
        }

        private void TrainExample(double[] input, double[] expectedOutput)
        {
            var output = Forward(input);
            System.Diagnostics.Debug.WriteLine("Cost: " + Cost(output, expectedOutput));
            BackPropagation(expectedOutput);
        }

        private void ResetDerivatives()
        {
            for (int i = 0; i < Layers.Length; i++)
            {
                for (int j = 0; j < Layers[i].Neurons.Length; j++)
                {
                    for (int k = 0; k < Layers[i].Neurons[j].DCDW.Length; k++)
                    {
                        Layers[i].Neurons[j].DCDW[k] = 0;
                    }
                    Layers[i].Neurons[j].DCDB = 0;
                }
            }
        }
    }
}
