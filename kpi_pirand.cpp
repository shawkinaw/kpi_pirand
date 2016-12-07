/******************************************************************************
  Copyright 2016
  All rights reserved.
  KNUPATH Proprietary and Confidential

  File: kpi_pirand.cpp

  Description: Host code for KPI pi estimation example using Monte Carlo method

******************************************************************************/

#include "kpi/kpi_runtime.hpp"
#include <iostream>
#include <fstream>
#include <ctime>

int main(int argc, char* argv[])
{
  // Get start of entire runtime
  std::clock_t begin_run = std::clock();

  // Create a context for complex 0 (the default)
  kpi::Context        ctx;

  // Create a command queue to submit tasks to the complex
  kpi::CommandQueue   cq = ctx.CreateCommandQueue();

  // Allocate a cluster block for the entire complex
  kpi::ClusterBlock   cb = ctx.AllocClusterBlock();

  // Write to an output file
  std::ofstream output;
  output.open("example_output.csv");

  // Output Info
  output << "Operating on " << cb.Size() << " clusters." << std::endl;
  output << "Iteration, coordsPerRank, # Samples,  Time,  Pi Value" << std::endl;

  // Randomizes how many increments of the power of 2 to run
  srand(time(NULL)); // Seed the time
  int max = 31; // 2^30 max power of 2 for int coordsPerRank
  int min = 2; // Minimum, do 1 run
  int runLimit = rand()%(max-min + 1) + min;

  // Run in increments of the power of 2
  for (int i=1;i<runLimit;i+=1)
  {
    // Get start of each iteration
    std::clock_t begin = std::clock();

    // Calculates # of coordinates per rank
    int coordsPerRank = 1 << i;

    // Load a kernel and set the argument to the coordsPerRank
    kpi::KernelWithArgs kwa = ctx.CreateKernelWithArgs("piRandRatio", coordsPerRank);

    // Submit the launch configuration to the queue. Use the returned
    // host connection to communicate with the kernels during while
    // the launch is running.
    kpi::HostConn hc =  cq.SubmitWithHostConn(
                          ctx.CreateLaunchConfig(
                            cb.Size(),
				                    kpi::ProcGroup(cb),
                            kwa),
                           cb);

    // Wait to receive a flit from RANK 0
    std::vector<float> pi_estimate(1);
    hc.ReceiveCount(pi_estimate);

    // Output Run Info
    output << i << ", " << cb.Size() << ", " << coordsPerRank << ",  " << double(coordsPerRank)*cb.Size()*8 << ",  "
      << double(std::clock() - begin)/CLOCKS_PER_SEC << ",  " << pi_estimate[0] << std::endl;
  }

  // Write total time it took to run program
  output << std::endl << "Total Time: " << double(std::clock() - begin_run)/CLOCKS_PER_SEC << std::endl;

  // Close file and exit
  output.close();
  return 0;
}
