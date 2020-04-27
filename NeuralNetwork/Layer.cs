namespace NeuralNetwork
{
    public class Layer
    {
        public Neuron[] Neurons { get; set; }

        public Layer(int neuronsCount)
        {
            Neurons = new Neuron[neuronsCount];
            for (int i = 0; i < Neurons.Length; i++)
            {
                Neurons[i] = new Neuron();
            }
        }
    }
}
