#  ML_SnakeProject

A machine learning project where a neural network learns to play the classic Snake game through evolutionary algorithms.  The snake learns to navigate towards apples using a genetic algorithm approach with mutation-based learning.

##  Table of Contents
- [Overview](#overview)
- [How It Works](#how-it-works)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Neural Network Architecture](#neural-network-architecture)
- [Training Process](#training-process)
- [Configuration](#configuration)
- [Project Structure](#project-structure)

##  Overview

This project demonstrates how artificial intelligence can learn to play games through trial and error. A population of snakes is simulated simultaneously, with each snake controlled by its own neural network. Through evolutionary selection and mutation, the snakes gradually improve their ability to find and collect apples while avoiding walls. 

##  How It Works

The system uses: 
- **Neural Networks**: Each snake has a neural network that decides its movement based on environmental inputs
- **Genetic Algorithm**:  The best-performing snakes pass their "genes" (neural network weights) to the next generation
- **Mutation**: Small random changes to the weights help explore new strategies
- **Fitness Function**: Snakes are rewarded for collecting apples and punished for inefficiency

##  Features

- **Parallel Training**:  Simulates 50 snakes simultaneously for efficient learning
- **Real-time Visualization**: Watch the best snake play after training
- **Save/Load System**: Save trained models and load them for future use
- **Configurable Parameters**:  Easily adjust population size, mutation rate, and training cycles
- **Fitness Tracking**: Monitor average and best fitness scores across training cycles

##  Requirements

- .NET 6.0 or higher
- Windows/Linux/macOS with terminal support
- Visual Studio 2022 or Visual Studio Code (optional, for development)

##  Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/MrJaki/ML_SnakeProject.git
   cd ML_SnakeProject
   ```

2. **Navigate to the project directory**
   ```bash
   cd Snake/Snake
   ```

3. **Build the project**
   ```bash
   dotnet build
   ```

##  Usage

### Running the Project

```bash
dotnet run
```

### First-Time Setup

When you first run the program: 

1. You'll be prompted:  `If you would like to load previous data write l or if you want to learn it write t:`
2. Type `t` and press Enter to start training from scratch
3. The program will train for 1000 cycles (this may take several minutes)
4. After training, the best snake will be displayed playing the game
5. The trained model will be automatically saved

### Loading a Trained Model

1. Run the program again
2. Type `l` and press Enter when prompted
3. Make sure you have a valid data file at the path specified in the code (lines 11 and 289)

### Configuring File Paths

Before running, update the file paths in `Program.cs`:
- **Line 11**: Set the path for loading saved models
  ```csharp
  StreamReader reader = new StreamReader(@"C:\path\to\your\model.txt");
  ```
- **Line 289**: Set the path for saving trained models
  ```csharp
  StreamWriter writer = new StreamWriter(@"C:\path\to\save\model.txt");
  ```

##  Neural Network Architecture

### Input Layer (16 neurons)
The snake receives 16 inputs representing its environment: 

- **Inputs 0-3**: Boundary detection (wall collision flags)
  - Is snake at right edge? 
  - Is snake at left edge?
  - Is snake at bottom edge?
  - Is snake at top edge?

- **Inputs 4-7**:  Normalized position (0.0 to 1.0)
  - Distance from left wall
  - Distance from right wall
  - Distance from top wall
  - Distance from bottom wall

- **Inputs 8-11**: Relative apple direction (normalized)
  - Horizontal distance to apple (normalized)
  - Inverse horizontal distance
  - Vertical distance to apple (normalized)
  - Inverse vertical distance

- **Inputs 12-15**:  Binary apple direction
  - Is apple to the left?
  - Is apple to the right?
  - Is apple above?
  - Is apple below?

### Hidden Layer (16 neurons)
- Fully connected to input layer
- Uses sigmoid activation function
- Includes bias terms

### Output Layer (4 neurons)
- One neuron for each direction:  Right, Left, Down, Up
- The direction with the highest activation is chosen
- Prevents 180-degree turns (moving directly backward)

##  Training Process

### Population-Based Learning
1. **Initialization**: 50 snakes spawn with random neural network weights
2. **Simulation**: Each snake plays simultaneously for up to 1000 steps
3. **Evaluation**:  Snakes earn fitness points: 
   - +100 points for collecting an apple
   - +1 point per step survived
   - -50 points if no apple collected within 50 steps
4. **Selection**:  The best-performing snake is identified
5. **Reproduction**: All snakes copy the best snake's weights
6. **Mutation**: 49 snakes receive small random weight changes (5% mutation rate)
7. **Repeat**: Process continues for 1000 training cycles

### Fitness Function
```
Fitness = (Apples Collected × 100) + Steps Survived - Penalties
```

##  Configuration

Key parameters you can modify in `Program.cs`:

| Parameter | Line | Default | Description |
|-----------|------|---------|-------------|
| `snakes` | 13 | 50 | Number of snakes in population |
| `trainingCycles` | 16 | 1000 | Number of training iterations |
| `maxSteps` | 76 | 1000 | Maximum steps per cycle |
| `mutationRate` | 253 | 0.05 | Probability of weight mutation (5%) |
| Grid width | 71, 286 | 26 | Width of game board |
| Grid height | 72, 287 | 16 | Height of game board |

##  Project Structure

```
ML_SnakeProject/
│
├── README.md                    # Project documentation
└── Snake/
    ├── Snake.sln                # Visual Studio solution file
    └── Snake/
        ├── Program.cs           # Main application code
        ├── Snake.csproj         # C# project file
        ├── bin/                 # Compiled binaries
        └── obj/                 # Intermediate build files
```

##  Game Display

After training, the visualization shows:
- `O` = Snake head position
- `A` = Apple position
- `.` = Empty space
- Score counter at the bottom

##  Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests
- Improve documentation

##  License

This project is open source and available for educational purposes.

##  Author

**MrJaki**
- GitHub: [@MrJaki](https://github.com/MrJaki)

##  Acknowledgments

- Inspired by classical genetic algorithms and neuroevolution
- Built with C# and .NET
- Console-based visualization for simplicity

---

**Note**: Training performance depends on your system.  Initial training may take 5-15 minutes. The snake typically learns to collect 5-10 apples consistently after full training. 

**PS**: Code was built by author, readme file and comments were mostly generated by GitHub Copilot.
