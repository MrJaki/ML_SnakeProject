using System;
using System.Threading;

namespace Snake
{
    internal class Program
    {
        static void Main(string[] args)
        {

            StreamReader reader = new StreamReader(@"");

            int snakes = 50;
            int snakesAliveCount = snakes;
            bool[] snakesAlive = new bool[snakes];
            int trainingCycles = 1000;
            double[,] snakesPosition = new double[2, snakes];
            double[] fitness = new double[snakes];
            int[] snakesDirection = new int[snakes];
            int[] stepsSinceLastApple = new int[snakes];

            double applePosx = 0;
            double applePosy = 0;

            double[] inputs = new double[16];
            double[] hiddenLayer = new double[16];
            double[] outputs = new double[4];

            double[,] weightsHidden = new double[snakes, inputs.Length * hiddenLayer.Length];
            double[,] weightsOutput = new double[snakes, outputs.Length * hiddenLayer.Length];
            double[,] bias = new double[snakes, hiddenLayer.Length];

            double[] bestWeightsHidden = new double[inputs.Length * hiddenLayer.Length];
            double[] bestWeightsOutput = new double[outputs.Length * hiddenLayer.Length];
            double[] bestBias = new double[hiddenLayer.Length];
            double bestFitness = 0;

            double[,] apples = new double[2, snakes];

            Random random = new Random();

            Console.Write("If you would like to load previus data write l or if you want to learn it write t: ");
            string input = Console.ReadLine();

            for (int j = 0; j < snakes; j++)
            {
                for (int i = 0; i < weightsHidden.GetLength(1); i++)
                    weightsHidden[j, i] = random.NextDouble() * 0.2 - 0.1;
                for (int i = 0; i < weightsOutput.GetLength(1); i++)
                    weightsOutput[j, i] = random.NextDouble() * 0.2 - 0.1;
                for (int i = 0; i < bias.GetLength(1); i++)
                    bias[j, i] = random.NextDouble() * 0.2 - 0.1;
            }

            if (input.ToLower() == "l") ReadData(ref weightsHidden, ref weightsOutput, ref bias, reader);

            reader.Close();

            for (int cycle = 0; cycle < trainingCycles; cycle++)
            {
                snakesAliveCount = snakes;
                Array.Fill(snakesAlive, true);
                Array.Fill(fitness, 0);
                Array.Fill(stepsSinceLastApple, 0);
                Array.Fill(snakesDirection, -1);

                for (int i = 0; i < snakes; i++)
                {
                    snakesPosition[0, i] = 10;
                    snakesPosition[1, i] = 10;
                    apples[0, i] = random.Next(0, 26);
                    apples[1, i] = random.Next(0, 16);
                }

                
                int maxSteps = 1000;
                for (int step = 0; step < maxSteps && snakesAliveCount > 1; step++)
                {
                    for (int i = 0; i < snakes; i++)
                    {
                        if (!snakesAlive[i]) continue;

                        SnakeBrain(snakesPosition[0, i], snakesPosition[1, i], ref inputs, apples[0, i], apples[1, i], snakesDirection[i]);
                        SnakesNeuralNetwork(inputs, hiddenLayer, ref outputs, ref weightsHidden, ref weightsOutput, ref bias, i);
                        SnakeMove(outputs, ref snakesPosition[0, i], ref snakesPosition[1, i], ref snakesDirection[i]);

                        snakesAlive[i] = IsSnakeAlive(snakesPosition[0, i], snakesPosition[1, i]);
                        if (!snakesAlive[i]) continue;

                        stepsSinceLastApple[i]++;

                        if (snakesPosition[0, i] == apples[0, i] && snakesPosition[1, i] == apples[1, i])
                        {
                            fitness[i] += 100;
                            GetApplePosition(ref applePosx, ref applePosy, snakesPosition[0, i], snakesPosition[1, i]);
                            apples[0, i] = applePosx;
                            apples[1, i] = applePosy;
                            stepsSinceLastApple[i] = 0;
                        }
                        else
                        {
                            fitness[i] += 1;
                            if (stepsSinceLastApple[i] > 50)
                            {
                                stepsSinceLastApple[i] = 0;
                                fitness[i] -= 50;
                            }
                        }
                    }
                }

                // Shranim najboljše vrednosti
                int bestSnake = Array.IndexOf(fitness, fitness.Max());
                if (fitness[bestSnake] > bestFitness)
                {
                    bestFitness = fitness[bestSnake];
                    for (int i = 0; i < bestWeightsHidden.Length; i++) bestWeightsHidden[i] = weightsHidden[bestSnake, i];
                    for (int i = 0; i < bestWeightsOutput.Length; i++) bestWeightsOutput[i] = weightsOutput[bestSnake, i];
                    for (int i = 0; i < bestBias.Length; i++) bestBias[i] = bias[bestSnake, i];
                }

                for (int j = 0; j < snakes; j++)
                {
                    for (int i = 0; i < weightsHidden.GetLength(1); i++) weightsHidden[j, i] = bestWeightsHidden[i];
                    for (int i = 0;i < weightsOutput.GetLength(1);i++) weightsOutput[j, i] = bestWeightsOutput[i];
                    for (int i = 0; i < bias.GetLength(1); i++) bias[j, i] = bestBias[i];
                }

                for (int i = 1; i < weightsHidden.GetLength(0); i++)
                {
                    Mutate(ref weightsHidden, ref weightsOutput, ref bias, random, i);
                }

                double avgFitness = 0;
                for (int j = 0; j < fitness.Length; j++)
                {
                    avgFitness += fitness[j];
                }

                Console.WriteLine($"Cycle {cycle + 1} complete. Best Fitness: {fitness[bestSnake]:F2}. Avg Fitness: {avgFitness/snakes}");
            }

            Console.Clear();
            Console.WriteLine("Training complete! Running best snake...");
            Thread.Sleep(1000);

            RunBestSnake(bestWeightsHidden, bestWeightsOutput, bestBias);
        }

        static void SnakeBrain(double posx, double posy, ref double[] inputs, double applePosx, double applePosy, double currentDirection)
        {
            for (int i = 0; i < inputs.Length; i++) inputs[i] = 0;

            inputs[0] = posx >= 25 ? 1 : 0;
            inputs[1] = posx <= 0 ? 1 : 0;
            inputs[2] = posy >= 15 ? 1 : 0;
            inputs[3] = posy <= 0 ? 1 : 0;  

            inputs[4] = posx / 25.0;
            inputs[5] = (25 - posx) / 25.0;
            inputs[6] = posy / 15.0;
            inputs[7] = (15 - posy) / 15.0;

            double deltaX = applePosx - posx;
            double deltaY = applePosy - posy;

            inputs[8] = (deltaX / 25.0 + 1) / 2.0;
            inputs[9] = (-deltaX / 25.0 + 1) / 2.0;
            inputs[10] = (deltaY / 15.0 + 1) / 2.0;
            inputs[11] = (-deltaY / 15.0 + 1) / 2.0;

            inputs[12] = posx > applePosx ? 1 : 0;
            inputs[13] = posx < applePosx ? 1 : 0;
            inputs[14] = posy > applePosy ? 1 : 0;
            inputs[15] = posy < applePosy ? 1 : 0;
        }

        static void SnakesNeuralNetwork(double[] inputs, double[] hiddenLayer, ref double[] outputs, ref double[,] weightsHidden, ref double[,] weightsOutput, ref double[,] bias, int snakes)
        {
            for (int i = 0; i < hiddenLayer.Length; i++)
            {
                hiddenLayer[i] = 0;
                for (int j = 0; j < inputs.Length; j++)
                {
                    hiddenLayer[i] += inputs[j] * weightsHidden[snakes, i * inputs.Length + j];
                }
                hiddenLayer[i] += bias[snakes, i];
                hiddenLayer[i] = Sigmoid(hiddenLayer[i]);
            }

            for (int i = 0; i < outputs.Length; i++)
            {
                outputs[i] = 0;
                for (int j = 0; j < hiddenLayer.Length; j++)
                {
                    outputs[i] += hiddenLayer[j] * weightsOutput[snakes, i * hiddenLayer.Length + j];
                }
                outputs[i] = Sigmoid(outputs[i]);
            }
        }

        static double Sigmoid(double x)
        {
            return 1.0 / (1.0 + Math.Exp(-Math.Clamp(x, -500, 500)));
        }

        static void SnakeMove(double[] outputs, ref double posx, ref double posy, ref int currentDirection)
        {
            double[] adjustedOutputs = new double[outputs.Length];
            Array.Copy(outputs, adjustedOutputs, outputs.Length);

            if (currentDirection != -1)
            {
                int oppositeDirection = -1;
                switch (currentDirection)
                {
                    case 0: oppositeDirection = 1; break;
                    case 1: oppositeDirection = 0; break;
                    case 2: oppositeDirection = 3; break;
                    case 3: oppositeDirection = 2; break;
                }
                adjustedOutputs[oppositeDirection] = double.MinValue;
            }

            int maxIndex = 0;
            double maxOutput = adjustedOutputs[0];

            for (int i = 1; i < adjustedOutputs.Length; i++)
            {
                if (adjustedOutputs[i] > maxOutput)
                {
                    maxOutput = adjustedOutputs[i];
                    maxIndex = i;
                }
            }

            switch (maxIndex)
            {
                case 0: posx += 1; break; 
                case 1: posx -= 1; break; 
                case 2: posy += 1; break; 
                case 3: posy -= 1; break; 
            }

            currentDirection = maxIndex;
        }

        static bool IsSnakeAlive(double posx, double posy)
        {
            return !(posx < 0 || posy < 0 || posx > 25 || posy > 15);
        }

        static void Mutate(ref double[,] weightsHidden, ref double[,] weightsOutput, ref double[,] bias, Random random, int index, double mutationRate = 0.05)
        {
            for (int i = 0; i < weightsHidden.GetLength(1) ; i++)
            {
                if (random.NextDouble() < mutationRate)
                    weightsHidden[index,i] += random.NextDouble() * 0.1 - 0.5;
            }
            for (int i = 0; i < weightsOutput.GetLength(1); i++)
            {
                if (random.NextDouble() < mutationRate)
                    weightsOutput[index,i] += random.NextDouble() * 0.1 - 0.5;
            }
            for (int i = 0; i < bias.GetLength(1); i++)
            {
                if (random.NextDouble() < mutationRate)
                    bias[index,i] += random.NextDouble() * 0.1 - 0.5;
            }
        }

        static void RunBestSnake(double[] weightsHidden, double[] weightsOutput, double[] bias)
        {
            double[] inputs = new double[16];
            double[] hiddenLayer = new double[16];
            double[] outputs = new double[4];

            double posX = 10;
            double posY = 10;
            int currentDirection = -1;

            double applePosx = 0;
            double applePosy = 0;

            Random random = new Random();
            applePosx = random.Next(0, 26);
            applePosy = random.Next(0, 16);

            StreamWriter writer = new StreamWriter(@"");

            WriteData(weightsHidden, weightsOutput, bias, writer);
            writer.Close();

            int score = 0;

            while (true)
            {
                Console.Clear();
                for (int y = 0; y < 16; y++)
                {
                    for (int x = 0; x < 26; x++)
                    {
                        if ((int)posX == x && (int)posY == y)
                            Console.Write("O");
                        else if ((int)applePosx == x && (int)applePosy == y)
                            Console.Write("A");
                        else
                            Console.Write(".");
                    }
                    Console.WriteLine();
                }

                Console.WriteLine($"Score: {score}");
                Thread.Sleep(700);

                SnakeBrain(posX, posY, ref inputs, applePosx, applePosy, currentDirection);
                SnakesNeuralNetworkRunning(inputs, hiddenLayer, ref outputs, ref weightsHidden, ref weightsOutput, ref bias);
                SnakeMove(outputs, ref posX, ref posY, ref currentDirection);

                if (!IsSnakeAlive(posX, posY))
                {
                    Console.WriteLine("Snake died 💀");
                    Console.ReadKey();
                    break;
                }

                if ((int)posX == (int)applePosx && (int)posY == (int)applePosy) score++;
                GetApplePosition(ref applePosx, ref applePosy, posX, posY);
            }
        }

        static void SnakesNeuralNetworkRunning(double[] inputs, double[] hiddenLayer, ref double[] outputs, ref double[] weightsHidden, ref double[] weightsOutput, ref double[] bias)
        {
            for (int i = 0; i < hiddenLayer.Length; i++)
            {
                hiddenLayer[i] = 0;
                for (int j = 0; j < inputs.Length; j++)
                {
                    hiddenLayer[i] += inputs[j] * weightsHidden[ i * inputs.Length + j];
                }
                hiddenLayer[i] += bias[ i];
                hiddenLayer[i] = Sigmoid(hiddenLayer[i]);
            }

            for (int i = 0; i < outputs.Length; i++)
            {
                outputs[i] = 0;
                for (int j = 0; j < hiddenLayer.Length; j++)
                {
                    outputs[i] += hiddenLayer[j] * weightsOutput[ i * hiddenLayer.Length + j];
                }
                outputs[i] = Sigmoid(outputs[i]);
            }
        }

        static void GetApplePosition(ref double applePosx, ref double applePosy, double posx, double posy)
        {
            Random random = new Random();   
            if (posx == applePosx && posy == applePosy)
            {
                applePosx = random.Next(0, 26);
                applePosy = random.Next(0, 16);
            }
        }

        static void ReadData(ref double[,] weightsInput, ref double[,] weightsOutput, ref double[,] bias, StreamReader reader)
        {
            for (int i = 0; i < weightsInput.GetLength(1); i++)
                weightsInput[0, i] = double.Parse(reader.ReadLine());

            for (int i = 0; i < weightsOutput.GetLength(1); i++)
                weightsOutput[0, i] = double.Parse(reader.ReadLine());

            for (int i = 0; i < bias.GetLength(1); i++)
                bias[0, i] = double.Parse(reader.ReadLine());
        }

        static void WriteData(double[] weightsInput, double[] weightsOutput, double[] bias, StreamWriter writer)
        {
            for (int i = 0; i < weightsInput.Length; i++)
                writer.WriteLine(weightsInput[i]);

            for (int i = 0; i < weightsOutput.Length; i++)
                writer.WriteLine(weightsOutput[i]);

            for (int i = 0; i < bias.Length; i++)
                writer.WriteLine(bias[i]);
        }
    }
}