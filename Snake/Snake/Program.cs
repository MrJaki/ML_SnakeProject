using System;
using System.Threading;

namespace Snake
{
    internal class Program
    {
        static void Main(string[] args)
        {
            // ===== FILE I/O SETUP =====
            // Configure the path to load previously trained neural network weights
            // IMPORTANT: Update this path to your desired file location before running
            StreamReader reader = new StreamReader(@"");

            // ===== TRAINING CONFIGURATION =====
            int snakes = 50;                            // Number of snakes in the population (genetic algorithm population size)
            int snakesAliveCount = snakes;              // Counter for alive snakes during simulation
            bool[] snakesAlive = new bool[snakes];      // Track which snakes are still alive
            int trainingCycles = 1000;                   // Number of generations/training iterations
            
            // ===== GAME STATE ARRAYS =====
            double[,] snakesPosition = new double[2, snakes];  // [0,i] = X position, [1,i] = Y position for snake i
            double[] fitness = new double[snakes];              // Fitness score for each snake
            int[] snakesDirection = new int[snakes];            // Current direction:  0=Right, 1=Left, 2=Down, 3=Up, -1=Initial
            int[] stepsSinceLastApple = new int[snakes];        // Steps taken since last apple collected (for penalty)

            // ===== APPLE POSITIONS =====
            double applePosx = 0;                       // Temporary apple X position
            double applePosy = 0;                       // Temporary apple Y position

            // ===== NEURAL NETWORK STRUCTURE =====
            double[] inputs = new double[16];           // Input layer:  16 neurons (environmental sensors)
            double[] hiddenLayer = new double[16];      // Hidden layer: 16 neurons (processing layer)
            double[] outputs = new double[4];           // Output layer: 4 neurons (one per direction)

            // ===== NEURAL NETWORK WEIGHTS =====
            // Weights connecting input layer (16) to hidden layer (16) = 16 * 16 = 256 weights per snake
            double[,] weightsHidden = new double[snakes, inputs.Length * hiddenLayer.Length];
            // Weights connecting hidden layer (16) to output layer (4) = 4 * 16 = 64 weights per snake
            double[,] weightsOutput = new double[snakes, outputs.Length * hiddenLayer.Length];
            // Bias values for each hidden layer neuron = 16 biases per snake
            double[,] bias = new double[snakes, hiddenLayer.Length];

            // ===== BEST SNAKE TRACKING =====
            // Store the weights of the best-performing snake for evolutionary selection
            double[] bestWeightsHidden = new double[inputs.Length * hiddenLayer.Length];
            double[] bestWeightsOutput = new double[outputs. Length * hiddenLayer.Length];
            double[] bestBias = new double[hiddenLayer.Length];
            double bestFitness = 0;

            // ===== APPLE POSITIONS FOR EACH SNAKE =====
            double[,] apples = new double[2, snakes];   // Each snake has its own apple position

            Random random = new Random();

            // ===== USER CHOICE:  LOAD OR TRAIN =====
            Console. Write("If you would like to load previus data write l or if you want to learn it write t: ");
            string input = Console.ReadLine();

            // ===== INITIALIZE NEURAL NETWORK WEIGHTS =====
            // Initialize all weights and biases with random values between -0.1 and 0.1
            for (int j = 0; j < snakes; j++)
            {
                // Initialize hidden layer weights
                for (int i = 0; i < weightsHidden.GetLength(1); i++)
                    weightsHidden[j, i] = random.NextDouble() * 0.2 - 0.1;
                
                // Initialize output layer weights
                for (int i = 0; i < weightsOutput. GetLength(1); i++)
                    weightsOutput[j, i] = random.NextDouble() * 0.2 - 0.1;
                
                // Initialize biases
                for (int i = 0; i < bias.GetLength(1); i++)
                    bias[j, i] = random.NextDouble() * 0.2 - 0.1;
            }

            // Load previously trained weights if user chose 'l'
            if (input. ToLower() == "l") ReadData(ref weightsHidden, ref weightsOutput, ref bias, reader);

            reader.Close();

            // ===== TRAINING LOOP =====
            // Run the genetic algorithm for the specified number of cycles
            for (int cycle = 0; cycle < trainingCycles; cycle++)
            {
                // ===== RESET FOR NEW CYCLE =====
                snakesAliveCount = snakes;
                Array.Fill(snakesAlive, true);          // All snakes start alive
                Array.Fill(fitness, 0);                 // Reset fitness scores
                Array.Fill(stepsSinceLastApple, 0);     // Reset step counters
                Array.Fill(snakesDirection, -1);        // Reset directions (no initial direction)

                // ===== INITIALIZE POSITIONS =====
                // Place all snakes at starting position (10, 10) with random apple positions
                for (int i = 0; i < snakes; i++)
                {
                    snakesPosition[0, i] = 10;          // Starting X position
                    snakesPosition[1, i] = 10;          // Starting Y position
                    apples[0, i] = random.Next(0, 26);  // Random apple X (0-25)
                    apples[1, i] = random.Next(0, 16);  // Random apple Y (0-15)
                }

                // ===== SIMULATION STEP LOOP =====
                // Run simulation for max steps or until only 1 snake remains
                int maxSteps = 1000;
                for (int step = 0; step < maxSteps && snakesAliveCount > 1; step++)
                {
                    // Process each snake
                    for (int i = 0; i < snakes; i++)
                    {
                        if (! snakesAlive[i]) continue;  // Skip dead snakes

                        // 1. Gather environmental inputs for the snake's neural network
                        SnakeBrain(snakesPosition[0, i], snakesPosition[1, i], ref inputs, apples[0, i], apples[1, i], snakesDirection[i]);
                        
                        // 2. Run neural network to decide movement direction
                        SnakesNeuralNetwork(inputs, hiddenLayer, ref outputs, ref weightsHidden, ref weightsOutput, ref bias, i);
                        
                        // 3. Move the snake based on neural network output
                        SnakeMove(outputs, ref snakesPosition[0, i], ref snakesPosition[1, i], ref snakesDirection[i]);

                        // 4. Check if snake is still alive (within boundaries)
                        snakesAlive[i] = IsSnakeAlive(snakesPosition[0, i], snakesPosition[1, i]);
                        if (!snakesAlive[i]) continue;

                        stepsSinceLastApple[i]++;

                        // ===== APPLE COLLECTION CHECK =====
                        if (snakesPosition[0, i] == apples[0, i] && snakesPosition[1, i] == apples[1, i])
                        {
                            // Snake collected an apple! 
                            fitness[i] += 100;                          // Large reward for apple
                            GetApplePosition(ref applePosx, ref applePosy, snakesPosition[0, i], snakesPosition[1, i]);
                            apples[0, i] = applePosx;
                            apples[1, i] = applePosy;
                            stepsSinceLastApple[i] = 0;                 // Reset step counter
                        }
                        else
                        {
                            // No apple collected this step
                            fitness[i] += 1;                            // Small reward for surviving
                            
                            // Penalty for inefficiency (wandering without finding apple)
                            if (stepsSinceLastApple[i] > 50)
                            {
                                stepsSinceLastApple[i] = 0;
                                fitness[i] -= 50;                       // Penalty for taking too long
                            }
                        }
                    }
                }

                // ===== EVOLUTIONARY SELECTION =====
                // Find the snake with the highest fitness (best performer)
                int bestSnake = Array.IndexOf(fitness, fitness. Max());
                
                // If this snake is better than all previous cycles, save its weights
                if (fitness[bestSnake] > bestFitness)
                {
                    bestFitness = fitness[bestSnake];
                    for (int i = 0; i < bestWeightsHidden.Length; i++) bestWeightsHidden[i] = weightsHidden[bestSnake, i];
                    for (int i = 0; i < bestWeightsOutput. Length; i++) bestWeightsOutput[i] = weightsOutput[bestSnake, i];
                    for (int i = 0; i < bestBias. Length; i++) bestBias[i] = bias[bestSnake, i];
                }

                // ===== REPRODUCTION =====
                // Copy the best snake's weights to all snakes (everyone becomes a clone of the best)
                for (int j = 0; j < snakes; j++)
                {
                    for (int i = 0; i < weightsHidden.GetLength(1); i++) weightsHidden[j, i] = bestWeightsHidden[i];
                    for (int i = 0;i < weightsOutput.GetLength(1);i++) weightsOutput[j, i] = bestWeightsOutput[i];
                    for (int i = 0; i < bias.GetLength(1); i++) bias[j, i] = bestBias[i];
                }

                // ===== MUTATION =====
                // Mutate all snakes except the first one (which remains the pure best)
                for (int i = 1; i < weightsHidden.GetLength(0); i++)
                {
                    Mutate(ref weightsHidden, ref weightsOutput, ref bias, random, i);
                }

                // ===== PROGRESS REPORTING =====
                // Calculate average fitness across all snakes
                double avgFitness = 0;
                for (int j = 0; j < fitness.Length; j++)
                {
                    avgFitness += fitness[j];
                }

                Console.WriteLine($"Cycle {cycle + 1} complete. Best Fitness: {fitness[bestSnake]: F2}.  Avg Fitness: {avgFitness/snakes}");
            }

            // ===== TRAINING COMPLETE =====
            Console. Clear();
            Console.WriteLine("Training complete! Running best snake...");
            Thread.Sleep(1000);

            // Run visualization of the best trained snake
            RunBestSnake(bestWeightsHidden, bestWeightsOutput, bestBias);
        }

        /// <summary>
        /// Creates input data for the neural network based on snake's current state
        /// This is the snake's "perception" of its environment
        /// </summary>
        /// <param name="posx">Snake's current X position</param>
        /// <param name="posy">Snake's current Y position</param>
        /// <param name="inputs">Array to fill with input values (16 inputs)</param>
        /// <param name="applePosx">Apple's X position</param>
        /// <param name="applePosy">Apple's Y position</param>
        /// <param name="currentDirection">Snake's current movement direction</param>
        static void SnakeBrain(double posx, double posy, ref double[] inputs, double applePosx, double applePosy, double currentDirection)
        {
            // Clear all inputs
            for (int i = 0; i < inputs.Length; i++) inputs[i] = 0;

            // ===== BOUNDARY DETECTION (Inputs 0-3) =====
            // Binary flags:  1 if at edge, 0 otherwise
            inputs[0] = posx >= 25 ? 1 : 0;     // At right edge? 
            inputs[1] = posx <= 0 ? 1 : 0;      // At left edge?
            inputs[2] = posy >= 15 ? 1 : 0;     // At bottom edge? 
            inputs[3] = posy <= 0 ? 1 : 0;      // At top edge?

            // ===== NORMALIZED POSITION (Inputs 4-7) =====
            // Values from 0.0 to 1.0 representing distance from edges
            inputs[4] = posx / 25.0;            // Distance from left (0 = left, 1 = right)
            inputs[5] = (25 - posx) / 25.0;     // Distance from right
            inputs[6] = posy / 15.0;            // Distance from top (0 = top, 1 = bottom)
            inputs[7] = (15 - posy) / 15.0;     // Distance from bottom

            // ===== RELATIVE APPLE DIRECTION (Inputs 8-11) =====
            // Normalized values indicating direction and distance to apple
            double deltaX = applePosx - posx;   // Positive = apple is to the right
            double deltaY = applePosy - posy;   // Positive = apple is below

            inputs[8] = (deltaX / 25.0 + 1) / 2.0;      // Horizontal distance (normalized 0-1)
            inputs[9] = (-deltaX / 25.0 + 1) / 2.0;     // Inverse horizontal distance
            inputs[10] = (deltaY / 15.0 + 1) / 2.0;     // Vertical distance (normalized 0-1)
            inputs[11] = (-deltaY / 15.0 + 1) / 2.0;    // Inverse vertical distance

            // ===== BINARY APPLE DIRECTION (Inputs 12-15) =====
            // Simple directional flags
            inputs[12] = posx > applePosx ? 1 : 0;      // Apple is to the left? 
            inputs[13] = posx < applePosx ? 1 : 0;      // Apple is to the right?
            inputs[14] = posy > applePosy ? 1 : 0;      // Apple is above?
            inputs[15] = posy < applePosy ?  1 : 0;      // Apple is below? 
        }

        /// <summary>
        /// Processes the neural network to determine snake movement
        /// Architecture: 16 inputs -> 16 hidden neurons -> 4 outputs
        /// </summary>
        /// <param name="inputs">Input layer values (16 neurons)</param>
        /// <param name="hiddenLayer">Hidden layer values (16 neurons)</param>
        /// <param name="outputs">Output layer values (4 neurons:  right, left, down, up)</param>
        /// <param name="weightsHidden">Weights from input to hidden layer</param>
        /// <param name="weightsOutput">Weights from hidden to output layer</param>
        /// <param name="bias">Bias values for hidden layer</param>
        /// <param name="snakes">Index of the current snake</param>
        static void SnakesNeuralNetwork(double[] inputs, double[] hiddenLayer, ref double[] outputs, ref double[,] weightsHidden, ref double[,] weightsOutput, ref double[,] bias, int snakes)
        {
            // ===== CALCULATE HIDDEN LAYER =====
            // Each hidden neuron is a weighted sum of all inputs, plus bias, through sigmoid
            for (int i = 0; i < hiddenLayer.Length; i++)
            {
                hiddenLayer[i] = 0;
                
                // Sum weighted inputs
                for (int j = 0; j < inputs.Length; j++)
                {
                    hiddenLayer[i] += inputs[j] * weightsHidden[snakes, i * inputs.Length + j];
                }
                
                // Add bias
                hiddenLayer[i] += bias[snakes, i];
                
                // Apply sigmoid activation function (squashes to 0-1 range)
                hiddenLayer[i] = Sigmoid(hiddenLayer[i]);
            }

            // ===== CALCULATE OUTPUT LAYER =====
            // Each output neuron is a weighted sum of all hidden neurons, through sigmoid
            for (int i = 0; i < outputs.Length; i++)
            {
                outputs[i] = 0;
                
                // Sum weighted hidden layer values
                for (int j = 0; j < hiddenLayer. Length; j++)
                {
                    outputs[i] += hiddenLayer[j] * weightsOutput[snakes, i * hiddenLayer.Length + j];
                }
                
                // Apply sigmoid activation
                outputs[i] = Sigmoid(outputs[i]);
            }
        }

        /// <summary>
        /// Sigmoid activation function
        /// Maps any value to range (0, 1) with smooth S-curve
        /// </summary>
        /// <param name="x">Input value</param>
        /// <returns>Value between 0 and 1</returns>
        static double Sigmoid(double x)
        {
            // Clamp x to prevent overflow in exponential function
            return 1.0 / (1.0 + Math.Exp(-Math.Clamp(x, -500, 500)));
        }

        /// <summary>
        /// Moves the snake based on neural network outputs
        /// Prevents 180-degree turns (moving directly backward)
        /// </summary>
        /// <param name="outputs">Neural network outputs (4 directions)</param>
        /// <param name="posx">Snake's X position (modified)</param>
        /// <param name="posy">Snake's Y position (modified)</param>
        /// <param name="currentDirection">Current direction (modified)</param>
        static void SnakeMove(double[] outputs, ref double posx, ref double posy, ref int currentDirection)
        {
            // Copy outputs to adjust them
            double[] adjustedOutputs = new double[outputs.Length];
            Array.Copy(outputs, adjustedOutputs, outputs.Length);

            // ===== PREVENT BACKWARD MOVEMENT =====
            // Snakes cannot move 180 degrees from current direction
            if (currentDirection != -1)
            {
                int oppositeDirection = -1;
                switch (currentDirection)
                {
                    case 0: oppositeDirection = 1; break;   // If moving right, can't move left
                    case 1: oppositeDirection = 0; break;   // If moving left, can't move right
                    case 2: oppositeDirection = 3; break;   // If moving down, can't move up
                    case 3: oppositeDirection = 2; break;   // If moving up, can't move down
                }
                adjustedOutputs[oppositeDirection] = double.MinValue;   // Make it impossible to choose
            }

            // ===== FIND HIGHEST OUTPUT =====
            // Choose the direction with the highest activation
            int maxIndex = 0;
            double maxOutput = adjustedOutputs[0];

            for (int i = 1; i < adjustedOutputs. Length; i++)
            {
                if (adjustedOutputs[i] > maxOutput)
                {
                    maxOutput = adjustedOutputs[i];
                    maxIndex = i;
                }
            }

            // ===== APPLY MOVEMENT =====
            // Move snake in chosen direction
            switch (maxIndex)
            {
                case 0: posx += 1; break;   // Move right
                case 1: posx -= 1; break;   // Move left
                case 2: posy += 1; break;   // Move down
                case 3: posy -= 1; break;   // Move up
            }

            // Update current direction
            currentDirection = maxIndex;
        }

        /// <summary>
        /// Checks if snake is within game boundaries
        /// </summary>
        /// <param name="posx">Snake's X position</param>
        /// <param name="posy">Snake's Y position</param>
        /// <returns>True if alive (within bounds), false if dead</returns>
        static bool IsSnakeAlive(double posx, double posy)
        {
            // Snake dies if it goes outside the 26x16 grid
            return !(posx < 0 || posy < 0 || posx > 25 || posy > 15);
        }

        /// <summary>
        /// Applies random mutations to neural network weights
        /// This introduces genetic variation for evolutionary learning
        /// </summary>
        /// <param name="weightsHidden">Hidden layer weights to mutate</param>
        /// <param name="weightsOutput">Output layer weights to mutate</param>
        /// <param name="bias">Bias values to mutate</param>
        /// <param name="random">Random number generator</param>
        /// <param name="index">Index of snake to mutate</param>
        /// <param name="mutationRate">Probability of mutating each weight (default 5%)</param>
        static void Mutate(ref double[,] weightsHidden, ref double[,] weightsOutput, ref double[,] bias, Random random, int index, double mutationRate = 0.05)
        {
            // ===== MUTATE HIDDEN LAYER WEIGHTS =====
            for (int i = 0; i < weightsHidden.GetLength(1) ; i++)
            {
                if (random.NextDouble() < mutationRate)
                    weightsHidden[index,i] += random.NextDouble() * 0.1 - 0.5;  // Small random change
            }
            
            // ===== MUTATE OUTPUT LAYER WEIGHTS =====
            for (int i = 0; i < weightsOutput.GetLength(1); i++)
            {
                if (random.NextDouble() < mutationRate)
                    weightsOutput[index,i] += random.NextDouble() * 0.1 - 0.5;
            }
            
            // ===== MUTATE BIASES =====
            for (int i = 0; i < bias.GetLength(1); i++)
            {
                if (random.NextDouble() < mutationRate)
                    bias[index,i] += random.NextDouble() * 0.1 - 0.5;
            }
        }

        /// <summary>
        /// Runs and visualizes the best trained snake
        /// Displays the game in console with real-time updates
        /// </summary>
        /// <param name="weightsHidden">Trained hidden layer weights</param>
        /// <param name="weightsOutput">Trained output layer weights</param>
        /// <param name="bias">Trained bias values</param>
        static void RunBestSnake(double[] weightsHidden, double[] weightsOutput, double[] bias)
        {
            // Initialize neural network layers
            double[] inputs = new double[16];
            double[] hiddenLayer = new double[16];
            double[] outputs = new double[4];

            // Starting position
            double posX = 10;
            double posY = 10;
            int currentDirection = -1;

            // Apple position
            double applePosx = 0;
            double applePosy = 0;

            // Spawn initial apple
            Random random = new Random();
            applePosx = random.Next(0, 26);
            applePosy = random.Next(0, 16);

            // ===== SAVE TRAINED MODEL =====
            // IMPORTANT: Update this path to your desired save location
            StreamWriter writer = new StreamWriter(@"");
            WriteData(weightsHidden, weightsOutput, bias, writer);
            writer. Close();

            int score = 0;

            // ===== GAME LOOP =====
            while (true)
            {
                // ===== RENDER GAME BOARD =====
                Console.Clear();
                for (int y = 0; y < 16; y++)
                {
                    for (int x = 0; x < 26; x++)
                    {
                        if ((int)posX == x && (int)posY == y)
                            Console.Write("O");         // Snake
                        else if ((int)applePosx == x && (int)applePosy == y)
                            Console.Write("A");         // Apple
                        else
                            Console.Write(".");         // Empty space
                    }
                    Console.WriteLine();
                }

                Console.WriteLine($"Score: {score}");
                Thread.Sleep(700);  // Control game speed (700ms per frame)

                // ===== SNAKE AI DECISION =====
                SnakeBrain(posX, posY, ref inputs, applePosx, applePosy, currentDirection);
                SnakesNeuralNetworkRunning(inputs, hiddenLayer, ref outputs, ref weightsHidden, ref weightsOutput, ref bias);
                SnakeMove(outputs, ref posX, ref posY, ref currentDirection);

                // ===== CHECK GAME OVER =====
                if (! IsSnakeAlive(posX, posY))
                {
                    Console.WriteLine("Snake died 💀");
                    Console.ReadKey();
                    break;
                }

                // ===== CHECK APPLE COLLECTION =====
                if ((int)posX == (int)applePosx && (int)posY == (int)applePosy) score++;
                GetApplePosition(ref applePosx, ref applePosy, posX, posY);
            }
        }

        /// <summary>
        /// Neural network processing for running the best snake (non-training mode)
        /// Same as SnakesNeuralNetwork but works with 1D weight arrays instead of 2D
        /// </summary>
        static void SnakesNeuralNetworkRunning(double[] inputs, double[] hiddenLayer, ref double[] outputs, ref double[] weightsHidden, ref double[] weightsOutput, ref double[] bias)
        {
            // Calculate hidden layer
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

            // Calculate output layer
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

        /// <summary>
        /// Generates a new apple position
        /// Called when snake collects an apple or at game start
        /// </summary>
        /// <param name="applePosx">Apple X position (modified)</param>
        /// <param name="applePosy">Apple Y position (modified)</param>
        /// <param name="posx">Snake's X position (to check if apple was collected)</param>
        /// <param name="posy">Snake's Y position (to check if apple was collected)</param>
        static void GetApplePosition(ref double applePosx, ref double applePosy, double posx, double posy)
        {
            Random random = new Random();   
            if (posx == applePosx && posy == applePosy)
            {
                // Spawn new apple at random position
                applePosx = random.Next(0, 26);
                applePosy = random.Next(0, 16);
            }
        }

        /// <summary>
        /// Loads previously saved neural network weights from file
        /// </summary>
        /// <param name="weightsInput">Array to load hidden layer weights into</param>
        /// <param name="weightsOutput">Array to load output layer weights into</param>
        /// <param name="bias">Array to load bias values into</param>
        /// <param name="reader">File stream reader</param>
        static void ReadData(ref double[,] weightsInput, ref double[,] weightsOutput, ref double[,] bias, StreamReader reader)
        {
            // Read hidden layer weights
            for (int i = 0; i < weightsInput.GetLength(1); i++)
                weightsInput[0, i] = double.Parse(reader.ReadLine());

            // Read output layer weights
            for (int i = 0; i < weightsOutput.GetLength(1); i++)
                weightsOutput[0, i] = double.Parse(reader.ReadLine());

            // Read biases
            for (int i = 0; i < bias.GetLength(1); i++)
                bias[0, i] = double.Parse(reader.ReadLine());
        }

        /// <summary>
        /// Saves trained neural network weights to file
        /// </summary>
        /// <param name="weightsInput">Hidden layer weights to save</param>
        /// <param name="weightsOutput">Output layer weights to save</param>
        /// <param name="bias">Bias values to save</param>
        /// <param name="writer">File stream writer</param>
        static void WriteData(double[] weightsInput, double[] weightsOutput, double[] bias, StreamWriter writer)
        {
            // Write hidden layer weights (one per line)
            for (int i = 0; i < weightsInput.Length; i++)
                writer.WriteLine(weightsInput[i]);

            // Write output layer weights
            for (int i = 0; i < weightsOutput.Length; i++)
                writer.WriteLine(weightsOutput[i]);

            // Write biases
            for (int i = 0; i < bias.Length; i++)
                writer.WriteLine(bias[i]);
        }
    }
}
