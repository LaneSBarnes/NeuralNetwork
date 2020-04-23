using System;
using NeuralNetwork;

namespace Tester
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Hello World!");

            var network = new Network(new int[] { 2, 3, 1 });

            Console.ReadKey();
        }
    }
}
