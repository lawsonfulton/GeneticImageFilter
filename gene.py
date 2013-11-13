import random
import cairo
from copy import deepcopy
import numpy as npy
import math
from time import time
import operator
import random as rand
import pickle

OBJECTIVE_IMAGE = "objective.png"
br,bg,bb,ba = 1.0, 1.0, 1.0, 1.0 #background colour

TWO_PI = 2.0 * math.pi

#A parameter emobodies an optimization variable
class Parameter:
    def __init__(self, min, max, initial_value = None):
        self.min = min
        self.max = max
        
        if not initial_value == None:
            self.val = initial_value
        else:
            self.val =  random.uniform(min,max)
    
    #pick a new uniform random number in the range, but scale it back by mut_fact
    def mutate(self, mut_fact):
        assert(mut_fact >= 0 and mut_fact <= 1)    
        # r = random.uniform(self.min, self.max)
        # d = r - self.val
        # self.val += d * mut_fact
        
        r = random.gauss(0.0, mut_fact)
        self.val += r

        # r = random.uniform(-mut_fact,mut_fact)
        # self.val += r
        if self.val > self.max:
            self.val = self.max
        elif self.val < self.min:
            self.val = self.min
        

#For now, gene consists of a circle and its parameters
class Gene:
    #Each of these should be a Parameter
    def __init__(self, x, y, radius, r, g, b, a):
        self.x = x
        self.y = y
        self.radius = radius
        self.r = r
        self.g = g
        self.b = b
        self.a = a
    
    #For each parameter, pick a new uniform random number in the range, but scale it back by mut_fact
    def mutate(self, mut_fact):
        self.x.mutate(mut_fact)
        self.y.mutate(mut_fact)
        self.radius.mutate(mut_fact)
        self.r.mutate(mut_fact)
        self.g.mutate(mut_fact)
        self.b.mutate(mut_fact)
        self.a.mutate(mut_fact)
    
class Genome:
    #Size is the number of genes in the genome
    #genes is an optional list of pre-created genes
    def __init__(self, size, genes = None):
        self.size = size
        if genes == None:
            self.genes = [ Gene( x = Parameter(0.0,1.0),
                                y      = Parameter(0.0,1.0),
                                radius = Parameter(0.00001,0.08),
                                r      = Parameter(0.0,1.0),
                                g      = Parameter(0.0,1.0),
                                b      = Parameter(0.0,1.0),
                                a      = Parameter(0.2,1.0)) for i in xrange(size)]
        else:
            self.genes = genes
            self.size = len(genes)
    
    #Mutate mutation_count random genes in the genome by a factor of mut_fact
    def mutate(self, mutation_count, mut_fact):
        assert(mutation_count >= 0 and mutation_count <= self.size)
        
        indices = random.sample(xrange(self.size), mutation_count)
        
        for i in indices:
            self.genes[i].mutate(mut_fact)
    
    #Creates a new genome where each gene has a recombination_fact probability that it comes from this genome
    #and a 1 - recombination_fact probability it comes from other_genome
    def recombinate_with(self, other_genome, recombination_fact):
        assert(recombination_fact >= 0 and recombination_fact <= 1)    
        assert(self.size == other_genome.size)
        
        new_genes = []
        for i in xrange(self.size):
            r = random.uniform(0,1)
            if r < recombination_fact:
                new_genes.append(self.genes[i])
            else:
                new_genes.append(other_genome.genes[i])
                
        return Genome(self.size, new_genes)
    
    #Save the graphical representation of the genome to a file
    def save_to_png(self, width, height, filename):
        gene_surf = cairo.ImageSurface (cairo.FORMAT_ARGB32, width, height)
        ctx = cairo.Context (gene_surf)
        ctx.scale (width, height)
        
        ctx.new_path()
        ctx.set_source_rgba(br,bg,bb,ba)
        ctx.rectangle(0,0,1,1) #Fill background colour
        ctx.fill()
        
        for gene in self.genes:
            ctx.new_path()
            ctx.set_source_rgba(gene.r.val, gene.g.val, gene.b.val, gene.a.val)  #colour of circle being drawn
            ctx.arc(gene.x.val, gene.y.val, gene.radius.val, 0.0, TWO_PI)
            ctx.fill()  # stroke current path
        
        gene_surf.write_to_png (filename)
        
        f = open('best.txt','w') 
        f.write(pickle.dumps(self))    
        
class GeneticAlgorithm:
    def __init__(self, objective_img_filename):
        #Data for the objective function
        self.obj_surf = cairo.ImageSurface.create_from_png(objective_img_filename)
        buf = self.obj_surf.get_data()
        self.obj_img_data = npy.frombuffer(buf, npy.uint8)
        format = self.obj_surf.get_format()
        self.width = self.obj_surf.get_width()
        self.height = self.obj_surf.get_height()
        
        self.gene_surf = cairo.ImageSurface (format, self.width, self.height)
        self.ctx = cairo.Context (self.gene_surf)
        self.ctx.scale (self.width, self.height)
    
    def optimize(self, max_iterations):
        #user parameters
        pop_size = 160
        genome_size = 800
        elite_fraction = 0.2
        mutation_count = 16#int(genome_size * 0.5)
        mut_fact = 0.04
        
        save_freq = 1
        #End of user parameters
        
        #dead_start = int(keep_fraction * pop_size) #What index to start throwing away indivduals in the sorted population
        
        population = [[Genome(genome_size), 0.0] for i in xrange(pop_size)] #The population consits of pairs of [genome, fitness]
        for p in population:
            p[1] = self.evaluate_fitness(p[0])
        
        population.sort(key = operator.itemgetter(1)) #Sort the population according to the fitness values
        
        iteration = 0
        done = False
        while not done:
            start_time = time()
            #mut_fact = float(iteration)/float(max_iterations)
            
            #reproduction
            new_pop = []
            for i in xrange(pop_size):
                rand_indices = rand.sample(xrange(0,int(elite_fraction * pop_size)), 2) #Two random individuals to reproduce
                
                baby_genome = population[rand_indices[0]][0].recombinate_with(population[rand_indices[1]][0], 0.5)
                baby_fitness = self.evaluate_fitness(baby_genome)
                
                new_pop.append([baby_genome, baby_fitness])
            
            population = new_pop
            
            #mutation stage adn recalc fitness
            for i in xrange(pop_size):
                new_genome = deepcopy(population[i][0])
                new_genome.mutate(mutation_count, mut_fact)
                
                new_fitness = self.evaluate_fitness(new_genome)
                
                if new_fitness < population[i][1]:
                    # #print str(new_fitness) + ", " + str(population[i][1])
                    population[i][0] = new_genome
                    population[i][1] = new_fitness
            
            #re-calculate fitness
            #for p in population:
            #    p[1] = self.evaluate_fitness(p[0])
                
            #Sort the population according to fitness
            #population = reversed(sorted(population, key = operator.itemgetter(1))) #Sort the population according to the fitness values
            population.sort(key = operator.itemgetter(1))
            #print "Took %f seconds to finish iteration %i with best fitness %f" %(time() - start_time, iteration, population[0][1] / 255.0)
            #for p in population:
            #    print str(p[1]) + ", ",
            #print
            
            #Save image
            if iteration % save_freq == 0:
                population[0][0].save_to_png(self.width,self.height,"iterations/" + str(iteration) + ".png")
                print "Took %f seconds to finish iteration %i with best fitness %f" %(time() - start_time, iteration, population[0][1] / 255.0)
            iteration += 1
            
            #check if done
            if iteration > max_iterations:
                done = True
            
            
        
        population[0][0].save_to_png(self.width,self.height,"Final.png")
        
        return population[0][1]

    #returns a measure of the fitness of a genome
    def evaluate_fitness(self, genome):
        self.ctx.new_path()
        self.ctx.set_source_rgba(br,bg,bb,ba)
        self.ctx.rectangle(0,0,1,1) #Fill background colour
        self.ctx.fill()
        
        for gene in genome.genes:
            self.ctx.new_path()
            self.ctx.set_source_rgba(gene.r.val, gene.g.val, gene.b.val, gene.a.val)  #colour of circle being drawn
            self.ctx.arc(gene.x.val, gene.y.val, gene.radius.val, 0.0, TWO_PI)
            self.ctx.fill()  # stroke current path
        
        buf = self.gene_surf.get_data()
        gene_img_data = npy.frombuffer(buf, npy.uint8)
        
        #d = npy.linalg.norm(gene_img_data - self.obj_img_data)     #Calculate the euclidian distance between the images for the fitness
        # d = 0.0
        n = float(gene_img_data.size)
        # for i in xrange(int(n)):
        # d += abs(int(gene_img_data[i]) - int(self.obj_img_data[i]))/n
        #print n
        a1 = npy.array(gene_img_data, dtype = float)
        a2 = npy.array(self.obj_img_data, dtype = float)
        d = npy.sum((a1 - a2) ** 2)

        return d
        
if __name__ == "__main__":
    start_time = time()
    
    alg = GeneticAlgorithm("objective.png")
    print alg.optimize(1000000)
    
    print "Completed in %fs" % (time() - start_time)