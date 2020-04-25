using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MathNet.Numerics;

namespace NeuralNetwork
{
    public class Network
    {
        public Random Random { get; set; }
        public Layer[] Layers { get; set; }

        public Network(int[] neuronCountPerLayer)
        {
            Layers = new Layer[neuronCountPerLayer.Length];
            for (int i = 0; i < Layers.Length; i++)
            {
                Layers[i] = new Layer(neuronCountPerLayer[i]);
            }

            Random = new Random(1234);
            for (int i = 0; i < Layers.Length - 1; i++)
            {
                for (int j = 0; j < Layers[i].Neurons.Length; j++)
                {
                    Layers[i].Neurons[j].Weights = new float[Layers[i + 1].Neurons.Length];
                    for (int k = 0; k < Layers[i].Neurons[j].Weights.Length; k++)
                    {
                        Layers[i].Neurons[j].Weights[k] = (float)(10 * Random.NextDouble() - 5);
                    }
                    Layers[i].Neurons[j].Bias = (float)(10 * Random.NextDouble() - 5);
                }
            }
            for (int j = 0; j < Layers[Layers.Length - 1].Neurons.Length; j++)
            {
                Layers[Layers.Length - 1].Neurons[j].Bias = (float)(10 * Random.NextDouble() - 5);
            }
        }

        /// <summary>
        /// Tells the network to guess the correct output given the input
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        public float[] Guess(float[] input)
        {
            throw new NotImplementedException();
            // Set input
            // Forward()
            // return output
        }

        /// <summary>
        /// Train the network with a single example
        /// </summary>
        /// <param name="input"></param>
        /// <param name="expectedOutput"></param>
        public void TrainExample(float[] input, float[] expectedOutput)
        {
            throw new NotImplementedException();
            // Set input
            // Forward()
            // return output
            // compute cost
            // BackPropagation()
        }

        //FIXME: make private
        public float[] Forward(float[] input)
        {
            // Set input into activations of first layer
            for (int j = 0; j < Layers[0].Neurons.Length; j++)
            {
                Layers[0].Neurons[j].Activation = input[j];
            }

            for (int i = 1; i < Layers.Length; i++)
            {
                for (int j = 0; j < Layers[i].Neurons.Length; j++) //3
                {
                    float sum = 0;
                    for (int k = 0; k < Layers[i - 1].Neurons.Length; k++) //2
                    {
                        sum += Layers[i - 1].Neurons[k].Weights[j] * Layers[i - 1].Neurons[k].Activation;
                    }
                    sum += Layers[i].Neurons[j].Bias;

                    Layers[i].Neurons[j].Activation = (float)SpecialFunctions.Logistic(sum);
                }
            }

            return Layers[Layers.Length - 1].Neurons.Select(x => x.Activation).ToArray();
        }

        //FIXME: make private static?
        public float Cost(float[] output, float[] expectedOutput)
        {
            float cost = 0;
            for (int i = 0; i < output.Length; i++)
            {
                cost += (float)Math.Pow(output[i] - expectedOutput[i], 2);
            }
            return cost;
        }

        private void BackPropagation()
        {
            throw new NotImplementedException();
        }
    }
}
