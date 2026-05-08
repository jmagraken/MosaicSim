import numpy as np
import cv2
import networkx as nx
import generativepy.color
import generativepy.geometry
from generativepy.drawing import make_image, setup
import scipy
import math
import jax.numpy as jnp
import jax
import pandas as pd
import jaxopt
import os
import networkx as nx
from argparse import ArgumentParser
from tqdm import tqdm

jax.config.update("jax_enable_x64", True)

d_SD = 1.3
d_DC = 0.65
D_DC = 1.3

EPSILON = 0.4

class Cone:
    '''
    A single cone or double cone member.
    '''

    def __init__(self, color: str, pos: np.ndarray, is_double: bool, id: int):
        """
        Initializes a Cone.

        Args:
            color (str): The cone colour. Either 'g' (green), 'b' (blue), or 'r' (red).
            pos (np.ndarray): The (x, y) position of the cone.
            is_double (bool): The cone is a double cone member if is_double == True. Otherwise,
                it is a single cone.
            id (int): A unique integer identifier for the cone.
        """
        self.color = color
        self.orig_pos = pos
        self.pos = pos
        self.is_double = is_double
        self.id=id
        self.doublable = self.color != 'b'

class DoubleCone:
    '''
    A double cone, comprised of two Cone objects.
    '''

    def __init__(self, cone1: Cone, cone2: Cone, centre: np.ndarray):
            """
            Initializes a DoubleCone.

            Args:
                cone1 (Cone): One member cone of the double cone.
                cone2 (Cone): The other member cone of the double cone. The ordering
                    of cone1 and cone2 has no effect.
                centre (np.ndarray): The geometic centre of the double cone, defined as
                    the average of the centres of cone1 and cone2.
            """
            self.cone1 = cone1
            self.cone2 = cone2
            self.centre = centre


class Mosaic:
    '''
    A cone mosaic, comprised of many single and double cones.

    The standard workflow is to instantiate a Mosaic object, call
    Mosaic.init_from_image(), then call Mosaic.execute().
    '''

    def __init__(self, delta: float, sigma: float, mu: float, tau: float):
        """
        Initializes a Mosaic.

        Args:
            delta (float): Maximum distance, in single cone diameters, that a cone
                may move from its initial position.
            sigma (float): The amount, in single cone diameters, by which the double
                cone domain is expanded radially in each expansion step.
            mu (float): The minimum permitted distance between two non-doublable cones
                after initialization, in single cone diameters.
            tau (float): The minimum permitted initial distance between two cones chosen
                to be non-doublable cones, in single cone diameters.
        """
        self.cones = []
        self.sc = []
        self.dc = []
        self.width = None
        self.height = None
        self.delta = delta
        self.sigma = sigma
        self.mu = mu
        self.tau = tau

            
    def init_from_image(self, src: str, compress='auto', max_cells=None):
        """
        Set up an initial single cone mosaic from a schematic ("stage 1").

        Automatically detects cone positions from a .jpg mosaic schematic
        using the Hough transform. The schematic must represent cones as
        circles of a solid color. Only supports red, green, and blue
        cones.

        Args:
            src (str): Path to mosaic schematic.
            compress (float or str, optional): Scaling factor of cell positions.
                Defaults to 'auto', which scales mosaic to attain the cell density
                of the perfect square mosaic.
            max_cells (int or None, optional): Trims the mosaic to contain max_cells
                cells. No action is taken if max_cells == None.
        """
        image = cv2.imread(src, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20, param1=50, param2=40, minRadius=5, maxRadius=50)
        circles = np.round(circles[0, :]).astype("int")
        mean_r = sum([circle[2] for circle in circles])/len(circles)
        points = np.array([[x*0.5/mean_r, y*0.5/mean_r] for (x, y, r) in circles])
        D_target = 5/(4*(d_SD**2-(d_DC/2)**2))
        factor = math.sqrt(Mosaic.get_density(points)/D_target) if compress == 'auto' else compress

        if max_cells:
            circles = np.array(sorted(circles, key=lambda x: max(abs(x[0]-image.shape[1]/2), abs(x[1]-image.shape[0]/2)))[:max_cells])

        x_min = min(circles[:,0])
        x_max = max(circles[:,0])
        y_min = min(circles[:,1])
        y_max = max(circles[:,1])

        self.width = (x_max-x_min) * 0.5/mean_r * factor + 5
        self.height = (y_max-y_min) * 0.5/mean_r * factor + 5

        count = 0

        for (x, y, r) in circles:
            bgr = image[y-1][x-1]
            if max(bgr) == bgr[0]:
                cone = Cone('b', np.array([(x-x_min)*0.5/mean_r*factor+2.5, (y-y_min)*0.5/mean_r*factor+2.5]), False, count)
                self.cones.append(cone)
                self.sc.append(cone)
            elif max(bgr) == bgr[2]:
                cone = Cone('r', np.array([(x-x_min)*0.5/mean_r*factor+2.5, (y-y_min)*0.5/mean_r*factor+2.5]), False, count)
                self.cones.append(cone)
                self.sc.append(cone)
            else:
                cone = Cone('g', np.array([(x-x_min)*0.5/mean_r*factor+2.5, (y-y_min)*0.5/mean_r*factor+2.5]), False, count)
                self.cones.append(cone)
                self.sc.append(cone)
            count += 1
        
        G = nx.Graph()
        G.add_nodes_from(self.cones)
        for i in range(len(self.cones)):
            for j in range(i+1, len(self.cones)):
                if not (self.cones[i].color == 'b' and self.cones[j].color == 'b') and np.linalg.norm(self.cones[i].orig_pos - self.cones[j].orig_pos) < self.tau:
                    G.add_edge(self.cones[i], self.cones[j])
        blues = [cone for cone in self.sc if cone.color == 'b']
        non_doublables = list(set(nx.maximal_independent_set(G, nodes=blues)) - set(blues))
        while len(non_doublables) < max(round((len(self.sc)/5-len(blues))), 0):
            non_doublables = list(set(nx.maximal_independent_set(G, nodes=blues)) - set(blues))

        for cone in non_doublables[:max(round((len(self.sc)/5-len(blues))), 0)]:
            cone.doublable = False


    def make_doubles(self, radius: float):
        """
        Forms double cones from neighbouring doublable single cones within
        the double cone domain ("stage 2").

        Args:
            radius (float): The current radius of the double cone domain.
        """
        doublables = [cone for cone in self.sc if cone.doublable and np.linalg.norm(cone.orig_pos - np.array([self.width/2, self.height/2])) <= radius]
        edges = []
        for i in range(len(doublables)):
                for j in range(i+1, len(doublables)):
                    if np.linalg.norm(doublables[i].orig_pos - doublables[j].orig_pos) - 2*self.delta <= d_DC:
                        edges.append((doublables[i], doublables[j], np.linalg.norm(doublables[i].pos-doublables[j].pos)))
        matching = []
        used_nodes = set()
        dist_key = lambda x: x[2]
        edges = sorted(edges, key = dist_key)
        for edge in edges:
            if edge[0] not in used_nodes and edge[1] not in used_nodes:
                matching.append(edge)
                used_nodes.add(edge[0])
                used_nodes.add(edge[1])
        for cone1, cone2, _ in matching:
            cone1.is_double = True
            cone2.is_double = True
            self.sc.remove(cone1)
            self.sc.remove(cone2)
            cone1_to_cone2 = cone2.pos - cone1.pos
            centre = cone1_to_cone2/2 + cone1.pos
            cone2.pos = centre + (cone1_to_cone2/np.linalg.norm(cone1_to_cone2)) * d_DC/2
            cone1.pos = centre + -1*(cone1_to_cone2/np.linalg.norm(cone1_to_cone2)) * d_DC/2
            self.dc.append(DoubleCone(cone1, cone2, centre))

    def optimize(self, radius: float, stochastic=True, verbose=False):
        """
        Minimizes the energy function, f, and updates the mosaic to the minimizer ("stage 3").

        Performs projected gradient descent on the energy function, f. If
        stochastic == True, performs projected gradient descent 100 times, starting
        from different samples in cone position domains and taking the minimum to be
        the least of the 100 local minima. Implements vectorized operations for
        efficiency, contrary to written descriptions of the energy function.

        Args:
            radius (float): Current radius of the double cone domain.
            stoachastic (bool, optional): Performs 100-start projected gradient descent if
                stochastic == True. Otherwise, performs single-start projected gradient descent.
            verbose (bool, optional): Verbose output if verbose == True.
        """
        x0 = []

        mask = []
        
        init_pos = []

        for cone in self.sc:
            if not stochastic or (stochastic and np.linalg.norm(cone.orig_pos - np.array([self.width/2, self.height/2])) > radius or np.linalg.norm(cone.orig_pos - np.array([self.width/2, self.height/2])) <= radius-self.sigma):
                x0.append(cone.pos[0])
                x0.append(cone.pos[1])
                mask.append(0)
                mask.append(0)
            else:
                x0.append(cone.orig_pos[0])
                x0.append(cone.orig_pos[1])
                mask.append(1)
                mask.append(1)
            init_pos.append(cone.orig_pos[0])
            init_pos.append(cone.orig_pos[1])
        for dc in self.dc:
            if not stochastic or (stochastic and np.linalg.norm(dc.cone1.orig_pos - np.array([self.width/2, self.height/2])) <= radius-self.sigma and np.linalg.norm(dc.cone2.orig_pos - np.array([self.width/2, self.height/2])) <= radius-self.sigma):
                x0.append(dc.centre[0])
                x0.append(dc.centre[1])
                mask.append(0)
                mask.append(0)
            else:
                x0.append((dc.cone1.orig_pos[0]+dc.cone2.orig_pos[0])/2)
                x0.append((dc.cone1.orig_pos[1]+dc.cone2.orig_pos[1])/2)
                mask.append(1)
                mask.append(1)
            init_pos.append((dc.cone1.orig_pos[0]+dc.cone2.orig_pos[0])/2)
            init_pos.append((dc.cone1.orig_pos[1]+dc.cone2.orig_pos[1])/2)
        for dc in self.dc:
            ang = np.arctan2(dc.cone1.pos[1]-dc.centre[1], dc.cone1.pos[0]-dc.centre[0])
            x0.append(ang)
            if np.linalg.norm(dc.cone1.orig_pos - np.array([self.width/2, self.height/2])) <= radius-self.sigma and np.linalg.norm(dc.cone2.orig_pos - np.array([self.width/2, self.height/2])) <= radius-self.sigma:
                mask.append(0)
            else:
                mask.append(1)
            init_pos.append(ang)

        x0 = jnp.array(x0)
        init_pos = jnp.array(init_pos)
        mask = jnp.array(mask)

        sings = jnp.array([1.0 if not s.doublable and np.linalg.norm(s.orig_pos - np.array([self.width/2, self.height/2])) <= radius else 0.0 for s in self.sc])
        sing_mask = jnp.outer(sings, sings)
        sing_mask *= jnp.triu(jnp.ones_like(sing_mask), 1)
        mosaic_mask = jnp.outer(sings, jnp.ones(len(self.dc)))

        sing_dc_close_mask = jnp.array([[jnp.linalg.norm((self.dc[j].cone1.orig_pos+self.dc[j].cone2.orig_pos)/2 - self.sc[i].orig_pos) - 2*self.delta <= d_DC/2 + D_DC/2 + 0.5 for j in range(len(self.dc))] for i in range(len(self.sc))])
        sing_close_mask = jnp.array([[jnp.linalg.norm(self.sc[j].orig_pos - self.sc[i].orig_pos) - 2*self.delta <= 1 for j in range(len(self.sc))] for i in range(len(self.sc))])
        dc_close_mask = jnp.array([[jnp.linalg.norm((self.dc[j].cone1.orig_pos+self.dc[j].cone2.orig_pos)/2 - (self.dc[i].cone1.orig_pos+self.dc[i].cone2.orig_pos)/2) - 2*self.delta <= d_DC + D_DC for j in range(len(self.dc))] for i in range(len(self.dc))])

        sing_dc_int_mask = jnp.array([[jnp.linalg.norm((self.dc[j].cone1.orig_pos+self.dc[j].cone2.orig_pos)/2 - self.sc[i].orig_pos) - 2*self.delta <= 0 for j in range(len(self.dc))] for i in range(len(self.sc))])

        sing_space_mask = jnp.array([[jnp.linalg.norm(self.sc[j].orig_pos - self.sc[i].orig_pos) - 2*self.delta <= self.mu for j in range(len(self.sc))] for i in range(len(self.sc))])

        def f1(mat, rad1, rad2):
            space = rad1 + rad2
            return 2*(jnp.sqrt(mat)-(space)*jnp.ones_like(mat))**2 * jnp.heaviside((space)**2*jnp.ones_like(mat)-mat, 0)
        
        def f2(mat, rad1, rad2):
            a = 1/50
            b = 10
            return -a*jnp.ones_like(mat)/((b*(jnp.sqrt(mat)-(rad1+rad2)*jnp.ones_like(mat)))**2 + jnp.ones_like(mat))
        
        def f3(mat, space):
            return 2*(jnp.sqrt(mat)-(space)*jnp.ones_like(mat))**2 * jnp.heaviside((space)**2*jnp.ones_like(mat)-mat, 0)
        

        
        def f(x):
            sing_x = x[0:2*len(self.sc):2]
            sing_y = x[1:2*len(self.sc):2]
            cent_x = x[2*len(self.sc):2*(len(self.sc)+len(self.dc)):2]
            cent_y = x[2*len(self.sc)+1:2*(len(self.sc)+len(self.dc)):2]
            thetas = x[2*(len(self.sc)+len(self.dc)):]

            cone1_x = cent_x + d_DC/2 * jnp.cos(thetas)
            cone1_y = cent_y + d_DC/2 * jnp.sin(thetas)
            cone2_x = cent_x - d_DC/2 * jnp.cos(thetas)
            cone2_y = cent_y - d_DC/2 * jnp.sin(thetas)

            sing_d2 = (jnp.outer(sing_x, jnp.ones_like(sing_x)) - jnp.outer(jnp.ones_like(sing_x), sing_x))**2 + (jnp.outer(sing_y, jnp.ones_like(sing_y)) - jnp.outer(jnp.ones_like(sing_y), sing_y))**2
            c1_sing_d2 = (jnp.outer(sing_x, jnp.ones_like(cone1_x)) - jnp.outer(jnp.ones_like(sing_x), cone1_x))**2 + (jnp.outer(sing_y, jnp.ones_like(cone1_y)) - jnp.outer(jnp.ones_like(sing_y), cone1_y))**2
            c2_sing_d2 = (jnp.outer(sing_x, jnp.ones_like(cone2_x)) - jnp.outer(jnp.ones_like(sing_x), cone2_x))**2 + (jnp.outer(sing_y, jnp.ones_like(cone2_y)) - jnp.outer(jnp.ones_like(sing_y), cone2_y))**2
            c1_c2_d2 = (jnp.outer(cone1_x, jnp.ones_like(cone1_x)) - jnp.outer(jnp.ones_like(cone2_x), cone2_x))**2 + (jnp.outer(cone1_y, jnp.ones_like(cone1_y)) - jnp.outer(jnp.ones_like(cone2_y), cone2_y))**2
            c1_c1_d2 = (jnp.outer(cone1_x, jnp.ones_like(cone1_x)) - jnp.outer(jnp.ones_like(cone1_x), cone1_x))**2 + (jnp.outer(cone1_y, jnp.ones_like(cone1_y)) - jnp.outer(jnp.ones_like(cone1_y), cone1_y))**2
            c2_c2_d2 = (jnp.outer(cone2_x, jnp.ones_like(cone2_x)) - jnp.outer(jnp.ones_like(cone2_x), cone2_x))**2 + (jnp.outer(cone2_y, jnp.ones_like(cone2_y)) - jnp.outer(jnp.ones_like(cone2_y), cone2_y))**2

            sing_f1 = jnp.sum(f1(sing_d2+jnp.identity(sing_d2.shape[0]), 0.5, 0.5) * sing_close_mask * jnp.triu(jnp.ones_like(sing_d2), 1))
            c1_sing_f1 = jnp.sum(f1(c1_sing_d2, 0.5, D_DC/2) * sing_dc_close_mask)
            c2_sing_f1 = jnp.sum(f1(c2_sing_d2, 0.5, D_DC/2) * sing_dc_close_mask)
            c1_c2_f1 = jnp.sum(f1(c1_c2_d2, D_DC/2, D_DC/2) * dc_close_mask * (jnp.triu(jnp.ones_like(c1_c2_d2), 1) + jnp.tril(jnp.ones_like(c1_c2_d2), -1)))
            c1_c1_f1 = jnp.sum(f1(c1_c1_d2+jnp.identity(c1_c1_d2.shape[0]), D_DC/2, D_DC/2) * dc_close_mask * (jnp.triu(jnp.ones_like(c1_c1_d2), 1)))
            c2_c2_f1 = jnp.sum(f1(c2_c2_d2+jnp.identity(c2_c2_d2.shape[0]), D_DC/2, D_DC/2) * dc_close_mask * (jnp.triu(jnp.ones_like(c2_c2_d2), 1)))

            mosaic_f2_1 = jnp.sum(f2(c1_sing_d2, d_SD, 0) * mosaic_mask * sing_dc_int_mask)
            mosaic_f2_2 = jnp.sum(f2(c2_sing_d2, d_SD, 0) * mosaic_mask * sing_dc_int_mask)

            sing_sep = jnp.sum(f3(sing_d2+jnp.identity(sing_d2.shape[0]), self.mu) * sing_mask * sing_space_mask)

            return sing_f1 + c1_sing_f1 + c2_sing_f1 + c1_c2_f1 + c1_c1_f1 + c2_c2_f1 + mosaic_f2_1 + mosaic_f2_2 + sing_sep


        optimizer = jaxopt.ProjectedGradient(f, _project_circle, maxiter=100000, verbose=False, tol=0.001)
        end = len(x0) - len(self.dc)
        
        if stochastic:
            lhs = scipy.stats.qmc.LatinHypercube(len(x0))

            samples = lhs.random(n=100)

            samples[:,0:end:2] = np.sqrt(samples[:,0:end:2]) * self.delta
            samples[:,1:end:2] *= 2*math.pi
            samples[:,end:] *= math.pi
            for i in range(0, end, 2):
                rad = samples[:,i]
                theta = samples[:,i+1]
                samples[:,i] = rad*np.cos(theta)
                samples[:,i+1] = rad*np.sin(theta)
                
            
            ics = [x0 + mask * samples[i] for i in range(100)]
            
            results = [optimizer.run(init_params=x, hyperparams_proj=(init_pos, self.delta, end)) for x in (tqdm(ics) if verbose else ics)]

            best = results[np.argmin([f(res.params) for res in results])]
            x_best = best.params
        else:
            best = optimizer.run(init_params=x0, hyperparams_proj=(init_pos, self.delta, end))
            x_best = best.params
        

        sc_pos = [np.array([x_best[2*i], x_best[2*i+1]]) for i in range(len(self.sc))]
        dc_centre = [np.array([x_best[2*(len(self.sc)+i)], x_best[2*(len(self.sc)+i)+1]]) for i in range(len(self.dc))]
        dc_cone1_pos = [dc_centre[i] + d_DC/2 * np.array([math.cos(x_best[2*(len(self.sc)+len(self.dc))+i]), math.sin(x_best[2*(len(self.sc)+len(self.dc))+i])]) for i in range(len(self.dc))]
        dc_cone2_pos = [dc_centre[i] - d_DC/2 * np.array([math.cos(x_best[2*(len(self.sc)+len(self.dc))+i]), math.sin(x_best[2*(len(self.sc)+len(self.dc))+i])]) for i in range(len(self.dc))]
            
        for i in range(len(self.sc)):
            self.sc[i].pos = sc_pos[i]
        for i in range(len(self.dc)):
            self.dc[i].centre = dc_centre[i]
            self.dc[i].cone1.pos = dc_cone1_pos[i]
            self.dc[i].cone2.pos = dc_cone2_pos[i]

        return


    def get_next_row(self) -> list:
        """
        Yields next row of output CSV.

        Constructs and returns the next row of the output CSV file. Each row
        consists of the x and y positions of each cone and a list of
        double cones, each comprised of a two-tuple of cone identifiers.

        Returns:
            list: The next row of the output CSV.
        """
        row = [self.SMS(edge_threshold=2)]
        for cone in self.cones:
            row.append(cone.pos[0])
            row.append(cone.pos[1])
        doubles = [(dc.cone1.id, dc.cone2.id) for dc in self.dc]
        row.append(doubles)
        return row
    
    def execute(self, dir: str, verbose=False):
        """
        Simulates the retinal transformation of the Mosaic self.

        Transforms the Mosaic self by repeatedly expanding the double cone domain,
        forming new double cones, and minimizing the energy function, terminating once
        the initial positions of all cones are contained within the double cone domain.
        Saves schematics of the initial and final mosaics and a CSV file documenting the
        state of the mosaic after each iteration to directory dir.

        Args:
            dir (str): Path of directory in which to save output.
            verbose (bool, optional): Verbose output if verbose == True.
        """
        os.makedirs(dir, exist_ok=True)
        cols = ['SMS']
        for cone in self.cones:
            cols.append(f'Cone {cone.id} x {cone.color} {"doublable" if cone.doublable else "non-doublable"}')
            cols.append(f'Cone {cone.id} y {cone.color} {"doublable" if cone.doublable else "non-doublable"}')
        cols.append('Doubles')
        df = pd.DataFrame(columns=cols)
        df.loc[len(df)] = self.get_next_row()
        self.export(f'{dir}/initial.png')
        num_iter = math.ceil(max([np.linalg.norm(cone.orig_pos - np.array([self.width/2, self.height/2])) for cone in self.sc])/self.sigma)
        for i in range(1, num_iter+1):
            if verbose:
                print(f'----------Iteration {i} of {num_iter}----------')
            self.make_doubles(i * self.sigma)
            self.optimize(i * self.sigma, verbose=verbose)
            df.loc[len(df)] = self.get_next_row()
            jax.clear_caches()
        self.export(f'{dir}/final.png')
        df.to_csv(f'{dir}/res.csv', index=False)
    

    def export(self, filename: str):
        """
        Exports a .png schematic of Mosaic self.

        Represents each cone as a circle, with its colour corresponding to
        its spectral identity. Denotes the centre of each cone with a
        black dot.

        Args:
            filename (str): Filepath in which to save the schematic.
        """
        def draw(ctx, pixel_width, pixel_height, frame_no, frame_count):
            colordict = {'r': 'green', 'g': 'green', 'b': 'blue', 'u': 'purple'}
            setup(ctx, pixel_width, pixel_height, background=generativepy.color.Color(1))
            for cone in self.cones:
                generativepy.geometry.Circle(ctx).of_center_radius(100*cone.pos, 100*(D_DC/2 if cone.is_double else 0.5)).fill(generativepy.color.Color(colordict[cone.color]))
                generativepy.geometry.Circle(ctx).of_center_radius(100*cone.pos, 5).fill(generativepy.color.Color(0))
            for cone in [(dc.cone1, dc.cone2) for dc in self.dc]:
                generativepy.geometry.Line(ctx).of_start_end(100*(cone[0].pos), 100*(cone[1].pos)).stroke(pattern=generativepy.color.Color(0), line_width=2)
                
        make_image(filename, draw, round(100*self.width), round(100*self.height))

    def SMS(self, edge_threshold=2) -> float:
        """
        Computes the square mosaic score of the Mosaic self.

        Single cones within a distance of edge_threshold single cone diameters
        from the axis-aligned minimum bounding box containing the initial
        positions of all cones are discounted.

        Args:
            edge_threshold (float): Distance from bounding box within which
                single cones are discounted.
        
        Returns:
            float: The SMS of the Mosaic self.
        """
        if len(self.dc) < 4:
            return 0
        count = 0
        min_x = min([cone.orig_pos[0] for cone in self.cones]) + edge_threshold
        max_x = max([cone.orig_pos[0] for cone in self.cones]) - edge_threshold
        min_y = min([cone.orig_pos[1] for cone in self.cones]) + edge_threshold
        max_y = max([cone.orig_pos[1] for cone in self.cones]) - edge_threshold
        
        valid_sc = [sc for sc in self.sc if sc.orig_pos[0] >= min_x and sc.orig_pos[0] <= max_x and sc.orig_pos[1] >= min_y and sc.orig_pos[1] <= max_y]

        for sc in valid_sc:
            is_valid = True
            dc_dists = [np.linalg.norm(dc.centre - sc.pos) for dc in self.dc]
            dc_enum = list(enumerate(dc_dists))
            dc_enum = sorted(dc_enum, key=lambda x: x[1])
            for i in range(4):
                sc_to_cent = self.dc[dc_enum[i][0]].centre - sc.pos
                cent_to_cone1 = self.dc[dc_enum[i][0]].cone1.pos - self.dc[dc_enum[i][0]].centre
                angle = math.acos(max(min(np.dot(sc_to_cent, cent_to_cone1)/(np.linalg.norm(sc_to_cent) * np.linalg.norm(cent_to_cone1)), 1.0), -1.0))
                is_valid &= angle >= math.pi/2-EPSILON and angle <= math.pi/2+EPSILON
            if is_valid:
                count += 1
        return count/len(valid_sc)
    
    def get_density(points: np.ndarray, edge_threshold=2) -> float:
        """
        Computes the density of a point cloud.

        Discounts points within a distance of edge_threshold from the boundary
        of the axis-aligned minimum bounding box of the points. In this application,
        the point cloud is the set of cone centres.

        Args:
            points (np.ndarray): Point cloud array with dimensions (num_points, 2)
            edge_threshold: Distance from bounding box within which to discount points.

        Returns:
            float: The density of the point cloud.
        """
        min_x = min(points[:,0]) + edge_threshold
        max_x = max(points[:,0]) - edge_threshold
        min_y = min(points[:,1]) + edge_threshold
        max_y = max(points[:,1]) - edge_threshold
        count = 0
        for point in points:
            if point[0] >= min_x and point[0] <= max_x and point[1] >= min_y and point[1] <= max_y:
                count += 1
        area = (max_x-min_x)*(max_y-min_y)
        return count/area
    



def _project_circle(x: jnp.array, hyperparams: tuple) -> jnp.array:

    x0, delta, end = hyperparams

    diff = x - x0
    length = diff.shape[0]

    pad_len = (2 - (length % 2)) % 2
    padded = jnp.pad(diff, (0, pad_len))
    pairs = padded.reshape(-1, 2)

    norms = jnp.linalg.norm(pairs, axis=1) / delta
    scales_all = jnp.maximum(norms, 1.0)

    indices = jnp.arange(scales_all.shape[0])
    active = indices < (end // 2)
    scales = jnp.where(active, scales_all, 1.0)

    rep_pairs = jnp.repeat(scales, 2)
    divisor = rep_pairs[:length]

    return x0 + diff / divisor



def main():
    parser = ArgumentParser()

    parser.add_argument('--delta', type=float, default=2.0)
    parser.add_argument('--sigma', type=float, default=1.0)
    parser.add_argument('--mu', type=float, default=2.5)
    parser.add_argument('--tau', type=float, default=0.0)
    parser.add_argument('--initial_mosaic', type=str, default='AH1', choices=['AH1', 'AH2', 'AH3', 'AH4', 'AH5',
                                                                              'Sab1', 'Sab2', 'Sab3', 'Sab4', 'Sab5'])
    parser.add_argument('--out_dir', type=str, default='output')
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()
    mosaic = Mosaic(args.delta, args.sigma, args.mu, args.tau)
    mosaic.init_from_image(f"InitialMosaics/{args.initial_mosaic}.jpg", compress='auto', max_cells=125)
    mosaic.execute(args.out_dir, verbose=args.verbose)



if __name__ == '__main__':
    main()