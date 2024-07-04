#include <iostream>
#include <stdio.h>

#include <string>
#include <vector>
#include <fstream>
#include <iterator>
#include <sstream>
#include <algorithm> 
#include <cctype>
#include <locale>
#include <omp.h>
#include "cnpy.h"

void read_emb_file_fast(std::string inp_file, std::string ids_file, std::string embs_file, int emb_dim)
{
    FILE * infile = fopen(inp_file.c_str(),"r");
    std::vector<float> embs;
    std::vector<int> ids;
    int id;
    float emb_val;
    while(fscanf(infile, "%d", &id) > 0)
    {
        for(int i=0; i<emb_dim; i++)
        {
            fscanf(infile, "%f", &emb_val);
            embs.push_back(emb_val);
        }
        ids.push_back(id);
    }
    std::vector<size_t> shape_ids({ids.size()}), shape_embs({ids.size(),emb_dim});
    cnpy::npy_save(ids_file,&ids[0],shape_ids,"w");
    cnpy::npy_save(embs_file,&embs[0],shape_embs,"w");
}

void read_emb_file_parallel(std::string inp_file, std::string ids_file, std::string embs_file, int emb_dim)
{
    std::ifstream infile(inp_file);

    int num_lines = 0; std::string line;
    while(std::getline(infile, line)){num_lines++;}
    infile.clear();
    infile.seekg(0);

    std::vector<float> embs;
    std::vector<int> ids;

    #pragma omp parallel
    {
        std::string line;
        int id;
        std::vector<float> emb(emb_dim);
        #pragma omp for
        for(int line_idx=0; line_idx<num_lines; line_idx++)
        {
            #pragma omp critical
            {
                std::getline(infile, line);
            }
            std::stringstream ss(line);
            ss >> id;
            for(int i=0;i<emb_dim;i++)
                ss >> emb[i];
            #pragma omp critical
            {
                ids.push_back(id);
                for(int i=0;i<emb_dim;i++)
                    embs.push_back(emb[i]);
            }
        }
    }
    std::vector<size_t> shape_ids({ids.size()}), shape_embs({ids.size(),emb_dim});
    cnpy::npy_save(ids_file,&ids[0],shape_ids,"w");
    cnpy::npy_save(embs_file,&embs[0],shape_embs,"w");
}
    

int main(int argc, char** argv)
{
    std::string inp_file(argv[1]), ids_file(argv[2]), embs_file(argv[3]);
    int emb_dim = std::stoi(argv[4]);
    read_emb_file_parallel(inp_file, ids_file, embs_file, emb_dim);
}