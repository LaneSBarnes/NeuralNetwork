using System;
using System.Collections.Generic;
using NeuralNetwork;

namespace Tester
{
    class Program
    {
        static void Main(string[] args)
        {
            var network = new Network(new int[] { 1, 1, 1, 1});

            List<(double[], double[])> examples = new List<(double[], double[])>();
            for (int i = 0; i < 1000; i++)
            {
                examples.AddRange(new List<(double[], double[])>()
                {
                    (new double[] { 1 }, new double[] { 1 }),
                    (new double[] { 0 }, new double[] { 0 }),
                });
            }

            network.TrainExamples(examples.ToArray());

            double[] input = new double[] { 1 };
            double[] expectedOutput = new double[] { 1 };
            double[] output = network.Forward(input);
            double cost = network.Cost(output, expectedOutput);

            string inputString = "";
            foreach (var inputNeuron in input)
            {
                inputString += inputNeuron + ", ";
            }
            string outputString = "";
            foreach (var outputNeuron in output)
            {
                outputString += outputNeuron + ", ";
            }
            string expectedOutputString = "";
            foreach (var expectedOutputNeuron in expectedOutput)
            {
                expectedOutputString += expectedOutputNeuron + ", ";
            }

            Console.WriteLine("Input: " + inputString);
            Console.WriteLine("ExpectedOutput: " + expectedOutputString);
            Console.WriteLine("Output: " + outputString);
            Console.WriteLine("Cost: " + cost);

            Console.ReadKey();
        }
    }
}
