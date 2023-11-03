#include <cstring>
#include <fstream>
#include <iostream>

#include "pargeo/parseCommandLine.h"

#include "IO.h"
#include "dpc_sddp.h"

int main(int argc, char *argv[]) {
  std::ios_base::sync_with_stdio(0);
  pargeo::commandLine P(
      argc, argv,
      "-n{noiseCut} -d{depCut} -i{inFile} -k{densityK} -o{outFile}");
  double noiseCut = P.getOptionDoubleValue("-n", 0);
  double depCut = P.getOptionDoubleValue("-d", 0);
  int K = P.getOptionIntValue("-k", 6);

  char *iFile = P.getOptionValue("-i");
  std::string oFile = P.getOptionValue("-o", std::string(""));
  std::string dFile = P.getOptionValue("-decision", std::string(""));
  bool spatial_median = !P.getOption("-objmedian");
  bool randomize = P.getOption("-random")
  std::cout << "use " << (spatial_median? "spatial median" : "object median") << " split\n";

  double *data;
  std::size_t n;
  std::size_t dim;
  DPC::load_text_file(iFile, data, n, dim);

  if (dim == 2) {
    DPC::dpc_sddp<2>(data, oFile, dFile, n, K, noiseCut, depCut, spatial_median, randomize);
  } else if (dim == 128) {
    DPC::dpc_sddp<128>(data, oFile, dFile, n, K, noiseCut, depCut, spatial_median, randomize);
  } else if (dim == 784) {
    DPC::dpc_sddp<784>(data, oFile, dFile, n, K, noiseCut, depCut, spatial_median, randomize);
  } else if (dim == 1024) {
    DPC::dpc_sddp<1024>(data, oFile, dFile, n, K, noiseCut, depCut, spatial_median, randomize);
  } else {
    std::cout << "Unsupported dimension, dim = "<< dim << "\n";
    exit(1);
  }
  free(data);
}