using System;
using System.Linq;
using MathNet.Numerics;

namespace NeuralNetwork
{
    public class Network
    {
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
                    for (int k = 0; k < Layers[i].Neurons[j].Weights.Length; k++)
                    {
                        Layers[i].Neurons[j].Weights[k] = 10 * _random.NextDouble() - 5;
                    }
                    Layers[i].Neurons[j].Bias = 10 * _random.NextDouble() - 5;
                }
            }
        }

        /// <summary>
        /// Tells the network to guess the correct output given the input
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
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
            return cost;
        }

        //FIXME: make private
        public void BackPropagation(double[] expectedOutput)
        {
            var dC_da = 2 * (Layers[1].Neurons[0].Activation - expectedOutput[0]);
            var z = Layers[0].Neurons[0].Activation * Layers[1].Neurons[0].Weights[0] + Layers[1].Neurons[0].Bias;
            var da_dz = _actFuncDeriv(z);
            var dz_dw = Layers[0].Neurons[0].Activation;
            var dC_dw = dC_da * da_dz * dz_dw;

            dC_da = 2 * (Layers[1].Neurons[0].Activation - expectedOutput[0]);
            da_dz = _actFuncDeriv(z);
            var dC_db = dC_da * da_dz;

            Layers[1].Neurons[0].Weights[0] -= 5 * dC_dw;
            Layers[1].Neurons[0].Bias -= 5 * dC_db;
        }

        private void TrainExample(double[] input, double[] expectedOutput)
        {
            var output = Forward(input);
            System.Diagnostics.Debug.WriteLine("Cost: " + Cost(output, expectedOutput));
            BackPropagation(expectedOutput);
        }
    }
}
