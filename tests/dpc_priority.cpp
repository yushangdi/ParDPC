#include<iostream>
#include<fstream>
#include<algorithm>
#include<cstring>
#include<unordered_set>

#include "parlay/parallel.h"
#include "parlay/sequence.h"
#include "parlay/primitives.h"
#include "pargeo/parseCommandLine.h"
#include "pargeo/pointIO.h"
#include "pargeo/point.h"
#include "pargeo/getTime.h"
#include "pargeo/unionFind.h"
#include "priorityKdTree/psKdTree.h"
#include "kdTree/kdTree.h"

#include "parameter.h"


parlay::sequence<pointD> compute_densities(parlay::sequence<point>& ptrs){
	pargeo::kdTree::node<dim, point>* tree = 
		pargeo::kdTree::build<dim, point>(ptrs, true, 1);
	
	parlay::sequence<pointD> ptrDs(ptrs.size());

	size_t num_workers = parlay::num_workers();

	parlay::parallel_for(0, ptrs.size(), [&](size_t i){
		ptrDs[i] = pointD(ptrs[i].coords(), pargeo::kdTree::rangeCount(tree, ptrs[i], drange));
	}, 1);

	return ptrDs;
}




int main(int argc, char* argv[]) {
	std::ios_base::sync_with_stdio(0);

	pargeo::timer densityT, depT, linkageT;

	pargeo::commandLine P(argc, argv, "-r{radiu} -n{noiseCut} -d{depCut} -i{inFile} -o{outFile}");
	drange = P.getOptionDoubleValue("-r", 10);
	noiseCut = P.getOptionDoubleValue("-n", 0);
	depCut = P.getOptionDoubleValue("-d", 0);

	char* iFile = P.getOptionValue("-i");
	std::string oFile = P.getOptionValue("-o", std::string(""));

	std::cout << "num_thread: " << parlay::num_workers() << std::endl;

	parlay::sequence<point> ptrs = pargeo::pointIO::readPointsFromFile<point>(iFile);
	int n = ptrs.size();
	std::cout << "n: " << n << std::endl;
	
	densityT.start();
	parlay::sequence<pointD> ptrDs = compute_densities(ptrs);
	std::cout<< "density: "<<densityT.get_next()<<std::endl;

	ptrDs = parlay::filter(ptrDs, [&](pointD ptr){
		if(ptr.attribute >= noiseCut) return true;
		else return false;
	});
	n = ptrDs.size();

	depT.start();
	parlay::sequence<int> depPtr(n);
	parlay::stable_sort_inplace(ptrDs, pointD::attCompRev);
	parlay::sequence<pointD> sptrs(ptrDs);

	pargeo::psKdTree::tree<dim, pointD>* root = pargeo::psKdTree::build<dim, pointD>(ptrDs, true, 16);
	root->pargeo::psKdTree::node<dim, pointD>::initParallel();
	depPtr[0] = -1;
	parlay::parallel_for(0, n, [&](size_t i){
		pointD* ptr = root->NearestNeighborBounded(i);
		if(ptr && sptrs[i].distSqr(*ptr)<depCut*depCut)
			depPtr[i] = ptr->attribute;
		else
			depPtr[i] = -1;
	},1);
	std::cout<< "dependent: "<< depT.get_next()<<std::endl;

	linkageT.start();
	pargeo::unionFind<int> UF(n);
	parlay::parallel_for(0, n, [&](int i){
		if(depPtr[i] != -1)
			UF.link(i, depPtr[i]);
	});
	parlay::sequence<int> cluster(n);
	parlay::parallel_for(0, n, [&](int i){
		cluster[i] = UF.find(i);
	});
	std::cout<< "link:" <<linkageT.get_next()<<std::endl;

	if(oFile.size()>0){
		std::ofstream fout(oFile);
		std::unordered_set<int> unique_ids;
		for(int i=0;i<n;i++) unique_ids.insert(cluster[i]);
		fout<<n<<" "<<unique_ids.size()<<'\n';
		for(int i=0;i<n;i++) fout<<sptrs[i][0]<<" "<<sptrs[i][1]<<" "<<cluster[i]<<'\n';
		fout.close();
	}
}

