using System;
using System.Collections.Generic;

namespace NeuralNetwork
{
    public class Neuron
    {
        public float Activation { get; set; }

        public float[] Weights { get; set; }
        public float Bias { get; set; }

        public Layer PreviousLayer { get; set; }
    }
}
