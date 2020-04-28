namespace NeuralNetwork
{
    public class Neuron
    {
        public double Activation { get; set; }

        public double[] Weights { get; set; } = new double[0];
        public double[] DCDW { get; set; } = new double[0];
        public double Bias { get; set; }
        public double DCDB { get; set; }
    }
}
