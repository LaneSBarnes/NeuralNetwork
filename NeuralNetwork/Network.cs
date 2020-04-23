using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork
{
    public class Network
    {
        public Layer[] Layers { get; set; }

        public Network(int[] neuronCountPerLayer)
        {
            Layers = new Layer[neuronCountPerLayer.Length];
            for (int i = 0; i < Layers.Length; i++)
            {
                Layers[i] = new Layer(neuronCountPerLayer[i]);
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

        private void Forward()
        {
            throw new NotImplementedException();
        }

        private void BackPropagation()
        {
            throw new NotImplementedException();
        }
    }
}
