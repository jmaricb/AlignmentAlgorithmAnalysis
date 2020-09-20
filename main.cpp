//
//  main.cpp
//  Poravnator
//
//  Created by Josip Maric on 01/09/2020.
//  Copyright © 2020 Josip Maric. All rights reserved.
//

#include <iostream>
#include <algorithm>
#include <vector>
#include <iostream>
#include <fstream>

#include <xmmintrin.h>
#include <emmintrin.h>
#include <pmmintrin.h>
#include <tmmintrin.h>
#include <smmintrin.h>
#include <nmmintrin.h>

#define MATCH 5
#define MISSMATCH -5
#define INSERTION -4
#define DELETION -4

const __m128i vm = _mm_setr_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
const __m128i zero_vec = _mm_setzero_si128();

void traceback(std::string &target, std::string &query, std::string &cigar, int8_t **matrix) {
    unsigned long row = target.size();
    unsigned long column = query.size();
    
    while(row > 0 && column > 0) {
        
        std::cout << "row " << row << "  , column " << column << "  , " << target[row-1] << " " << query[column-1] << std::endl;
        
        if (matrix[row-1][column-1] >= matrix[row][column-1] && matrix[row-1][column-1] >= matrix[row-1][column]) {
            if (target[row-1] == query[column-1]) {
                cigar.append("M");
            } else {
                cigar.append("S");
            }
            row -= 1;
            column -= 1;
        } else if(matrix[row][column-1] > matrix[row-1][column]) {
            column -= 1;
            cigar.append("I");
        } else {
            row -= 1;
            cigar.append("D");
        }
    }
    
    while (row > 0) {
        row -= 1;
        cigar.append("D");
    }
    
    while (column > 0) {
        column -= 1;
        cigar.append("I");
    }
}

void classic_align(std::string &target, std::string &query) {
        
    int8_t **matrix = new int8_t* [target.size() + 1];
    
    for (int i = 0; i < target.size() + 1; i++) {
        matrix[i] = new int8_t [query.size() + 1];
    }
    
    for (int i = 0; i < target.size() + 1; i++) {
        matrix[i][0] = i * INSERTION;
    }
    
    for(int j = 0; j < query.size() + 1; j++) {
        matrix[0][j] = j * DELETION;
    }

    for (int i = 1; i < target.size() + 1; i++) {
        for (int j = 1; j < query.size() + 1; j++) {
            bool is_match = target[i-1] == query[j-1];
            int8_t diagonal_value = (is_match ? MATCH : MISSMATCH) + matrix[i-1][j-1];
            int8_t upper_value = matrix[i-1][j] + INSERTION;
            int8_t left_value = matrix[i][j-1] + DELETION;
            matrix[i][j] = std::max(std::max(upper_value, left_value), diagonal_value);
        }
    }

    for (int i = 0; i < target.size() + 1; i++) {
        delete[] matrix[i];
    }
    
    delete[] matrix;
}

void SSE_align(std::string &target, std::string &query) {
    int8_t **matrix = new int8_t* [target.size() + query.size() + 1 - 31];
    
    std::string query_rev = query;
    std::reverse(query_rev.begin(), query_rev.end());
    
    const char *reference_safe = target.c_str();
    const char *query_safe = query_rev.c_str();
    
    for (int i = 0; i < (target.size() + query.size() + 1) - 31; i++) {
        matrix[i] = new int8_t [(target.size() + query.size() + 1) - 31];
    }
        
    matrix[1][0] = DELETION;
    matrix[1][1] = INSERTION;
    
    __m128i match_vec = _mm_set1_epi8(MATCH);
    __m128i mismatch_vec = _mm_set1_epi8(MISSMATCH);
    __m128i deletion_vec = _mm_set1_epi8(DELETION);
    __m128i insertion_vec = _mm_set1_epi8(INSERTION);
        
    for (int row = 1; row < (target.size() + query.size() + 1) - 32; row++) {
        if (row < target.size() - 32) {
            matrix[row + 1][0] = (row + 1) * INSERTION;
        }
        
        int query_offset = (row >= target.size() - 32) ? (row - (target.size() - 32))  : 0;
        int query_size = (query.size() - 32) - query_offset;
        int limiter =  (row >= target.size() - 32) ? ((target.size() - 32) - query_size) :
            std::max((int)0, (int)(row - (query.size() - 32)));
        int ref_limit = (row >= target.size() - 32) ? target.size() - 32 : row;
        int i = ref_limit;
                
        while (i > limiter) {
            int j = ref_limit - i;
            i = i - 16;
            
            __m128i reference_chars = _mm_loadu_si128((__m128i *)&reference_safe[i + 16]);
            __m128i query_chars = _mm_loadu_si128((__m128i *)&query_safe[(query_size - (j + 16)) + 32]);
            
            __m128i mask = _mm_cmpeq_epi8(reference_chars, query_chars);
            __m128i z_vec = _mm_blendv_epi8 (mismatch_vec, match_vec, mask);
                        
            z_vec = _mm_shuffle_epi8(z_vec, vm);

            __m128i up_previous = _mm_loadu_si128((__m128i *)&matrix[row][j]);
            up_previous = _mm_add_epi8(up_previous, insertion_vec);

            __m128i left_previous = _mm_loadu_si128((__m128i *)&matrix[row][j + 1]);
            left_previous = _mm_add_epi8(left_previous, deletion_vec);

            int column = (row - 1 >= target.size() - 32) ? j + 1 : j;
            
            __m128i diagonal_previous = _mm_loadu_si128((__m128i *)&matrix[row - 1][column]);
            diagonal_previous =  _mm_add_epi8(diagonal_previous, z_vec);

            __m128i max = _mm_max_epi8 (left_previous, diagonal_previous);
            max = _mm_max_epi8 (up_previous, max);

            int target_column = (row  < target.size() - 32) ? j + 1 : j;
            _mm_storeu_si128((__m128i *)&matrix[row+1][target_column], max);
        }
        
        if(row < query.size() - 32) {
            matrix[row+1][row+1] = (row + 1) * DELETION;
        }
    }
    
    for (int i = 0; i < (target.size() + query.size() + 1) - 31; i++) {
        delete[] matrix[i];
    }

    delete[] matrix;
}

int main(int argc, const char * argv[]) {
    
    std::ifstream infile(argv[1]);
    std::vector<std::string> sequences;
    
    bool use_simd = true;
    
    std::string line;
    while (std::getline(infile, line)) {
        if (use_simd) {
            line.insert(0, "NNNNNNNNNNNNNNNN");
            line.append("NNNNNNNNNNNNNNNN");
        }
        sequences.emplace_back(line);
    }
    
    std::string *target;
    std::string *query;
    
    for (int k = 0; k < sequences.size() - 1; k++) {
        for (int l = 1; l < sequences.size(); l++) {
            if (k == l) {
                continue;
            }
            target = sequences[k].size() > sequences[l].size() ? &sequences[k] : &sequences[l];
            query = sequences[k].size() < sequences[l].size() ? &sequences[k] : &sequences[l];
            
            if (use_simd) {
                SSE_align(*target, *query);
            } else {
                classic_align(*target, *query);
            }
        }
    }
    
    return 0;
}

