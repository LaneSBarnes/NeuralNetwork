using System;
using NeuralNetwork;

namespace Tester
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Hello World!");

            var network = new Network(new int[] { 2, 3, 3, 2 });

            float[] input = new float[] { 1.0F, 0.0F };
            float[] output = network.Forward(input);

            string outputString = "";
            foreach (var outputNeuron in output)
            {
                outputString += outputNeuron + ", ";
            }

            Console.WriteLine("Output: " + outputString);
            Console.WriteLine("Cost: " + network.Cost(output, new float[] { 1, 0 }));

            Console.ReadKey();
        }
    }
}
