namespace NeuralNetwork
{
    public class Neuron
    {
        public double Activation { get; set; }

        public double[] Weights { get; set; }
        public double[] DCDW { get; set; }
        public double Bias { get; set; }
        public double DCDB { get; set; }
    }
}
