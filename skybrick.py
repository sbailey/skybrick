"""
skybrick
========

Code for calculating bricks and bricklets, which are a tiling of the
sky with the following properties:

- bricks form rows in dec like a brick wall; edges are constant RA or dec
- they are rectangular with longest edge shorter or equal to bricksize
- circles at the poles with diameter=bricksize
- there are an even number of bricks per row
- n x n bricklets form a brick, where n is even
"""

from __future__ import absolute_import, division, print_function

import copy
import numpy as np

class Bricks(object):
    """Bricks Object
    """
    def __init__(self, bricksize=1.0):
        """Create Bricks object such that all bricks have longest size < bricksize
        """
        #- The basic arrays to fill, starting with special case at south pole
        self.ra = [0.0, ]
        self.ra_min = [0.0, ]
        self.ra_max = [360.0, ]
        self.dec = [-90.0, ]
        self.dec_min = [-90.0, ]
        self.dec_max = [-90.0 + bricksize/2, ]
        self.row = [0, ]
        self.column = [0, ]
        self.kind = [0, ]
                
        #- fill in the bricks not at the poles
        dec_edges = np.arange(-90.0+bricksize/2, +90, bricksize)
        dec_centers = 0.5*(dec_edges[0:-1] + dec_edges[1:])
        for i, dec in enumerate(dec_centers):
            declo = np.abs(dec)-bricksize/2
            n = (360/bricksize * np.cos(declo*np.pi/180))
            ncol = int(np.ceil(n/2)*2)
            self.dec.extend( np.ones(ncol)*dec )
            self.dec_min.extend( np.ones(ncol)*dec_edges[i] )
            self.dec_max.extend( np.ones(ncol)*dec_edges[i+1] )
            ra_edges = np.linspace(0.0, 360.0, ncol+1)
            self.ra.extend(0.5*(ra_edges[0:-1] + ra_edges[1:]))
            self.ra_min.extend(ra_edges[0:-1])
            self.ra_max.extend(ra_edges[1:])
            self.row.extend( np.ones(ncol, dtype=int)*(i+1) )
            self.column.extend( np.arange(ncol, dtype=int) )
            #- rows alternqte bricks of type 0 & 1; odd rows have 2 & 3
            if (i+1)%2 == 0:
                self.kind.extend(np.arange(ncol)%2)     #- 0 1 0 1 0 1 ...
            else:
                self.kind.extend(2+np.arange(ncol)%2)   #- 2 3 2 3 2 3 ...

        #- special case at the north pole
        self.ra.append(0.0)
        self.ra_min.append(0.0)
        self.ra_max.append(360.0)
        self.dec.append(90.0)
        self.dec_min.append(90.0 - bricksize/2)
        self.dec_max.append(90.0)
        self.row.append(len(dec_centers)+1)
        self.column.append(0)
        if self.kind[-1] < 2:
            self.kind.append(2)
        else:
            self.kind.append(0)
            
        #- Convert to numpy arrays
        self.ra = np.array(self.ra)
        self.ra_min = np.array(self.ra_min)
        self.ra_max = np.array(self.ra_max)

        self.dec = np.array(self.dec)
        self.dec_min = np.array(self.dec_min)
        self.dec_max = np.array(self.dec_max)
        
        self.row = np.array(self.row)
        self.column = np.array(self.column)
        self.kind = np.array(self.kind)

    def __iter__(self):
        for i in range(len(self.ra_min)):
            yield Brick(self.ra_min[i], self.ra_max[i],
                        self.dec_min[i], self.dec_max[i],
                        self.kind[i],
                        self.row[i], self.column[i]
                        )
        
    def __getitem__(self, ii):
        result = copy.copy(self)
        result.ra = result.ra[ii]
        result.ra_min = result.ra_min[ii]
        result.ra_max = result.ra_max[ii]
        result.dec = result.dec[ii]
        result.dec_min = result.dec_min[ii]
        result.dec_max = result.dec_max[ii]
        result.row = result.row[ii]
        result.column = result.column[ii]
        result.kind = result.kind[ii]

        return result

class Brick(object):
    def __init__(self, ra_min, ra_max, dec_min, dec_max, kind, row, column):
        self.ra_min = ra_min
        self.ra_max = ra_max
        self.dec_min = dec_min
        self.dec_max = dec_max
        self.kind = kind
        self.row = row
        self.column = column
        
    def __repr__(self):
        return str((self.ra_min, self.ra_max, self.dec_min, self.dec_max, self.kind, self.row, self.column))

class Bricklets(object):
    def __init__(self, n, brick):
        """
        Divide a brick into n x n bricklets
        """
        b = brick   #- shorthand
        
        #- special case at south pole
        if (b.dec_min == -90.0) or (b.dec_max == 90.0):
            self.n = n
            nrow, ncol = n//2, n*2
            self.ntot = nrow*ncol
            edgesra = np.linspace(0, 360, ncol+1)
            ddec = 2*(b.dec_max - b.dec_min) / n
            if b.dec_min < 0:
                edgesdec = np.linspace(-90.0+ddec/2, b.dec_max, nrow + 1)
            else:
                edgesdec = np.linspace(b.dec_min, b.dec_max-ddec/2, nrow + 1)
                
            self.ra_min = np.tile(edgesra[0:ncol], nrow)
            self.ra_max = np.tile(edgesra[1:ncol+1], nrow)
            self.dec_min = np.repeat(edgesdec[0:nrow], ncol)
            self.dec_max = np.repeat(edgesdec[1:nrow+1], ncol)
            self.kind = np.zeros([nrow, ncol], dtype=int)
            for i in range(0,nrow-1,2):
                for j in range(0,ncol-1,2):
                    if n%2 == 0:
                        self.kind[i:nrow:2, j:ncol:2] = 0
                        self.kind[i:nrow:2, j+1:ncol:2] = 1
                        self.kind[i+1:nrow:2, j:ncol:2] = 2
                        self.kind[i+1:nrow:2, j+1:ncol:2] = 3
                    else:
                        self.kind[i:nrow:2, j:ncol:2] = 2
                        self.kind[i:nrow:2, j+1:ncol:2] = 3
                        self.kind[i+1:nrow:2, j:ncol:2] = 0
                        self.kind[i+1:nrow:2, j+1:ncol:2] = 1

            self.kind = self.kind.ravel()
            
            #- Reverse order of kind array at the north pole to fix the
            #- every other 0/1 vs. 2/3 phasing
            if b.dec_max == 90.0:
                self.kind = self.kind[-1::-1]
            
            #- the cap at the pole
            self.ra_min = np.concatenate([self.ra_min, [0.0,]])
            self.ra_max = np.concatenate([self.ra_max, [360.0,]])
            if b.dec_min == -90.0:
                self.dec_min = np.concatenate([self.dec_min, [-90.0,]])
                self.dec_max = np.concatenate([self.dec_max, [-90.0+ddec/2,]])
            else:
                self.dec_min = np.concatenate([self.dec_min, [90.0-ddec/2,]])
                self.dec_max = np.concatenate([self.dec_max, [90.0,]])
                
            if n%2 == 0:
                self.kind = np.concatenate([self.kind, [2, ]])
            else:
                self.kind = np.concatenate([self.kind, [0, ]])
            
            return
        
        self.n = n
        self.ntot = n*n
        
        edgesra = np.linspace(b.ra_min, b.ra_max, n+1)
        edgesdec = np.linspace(b.dec_min, b.dec_max, n+1)
        self.ra_min = np.tile(edgesra[0:n], n)
        self.ra_max = np.tile(edgesra[1:n+1], n)
        self.dec_min = np.repeat(edgesdec[0:n], n)
        self.dec_max = np.repeat(edgesdec[1:n+1], n)
        
        self.kind = np.zeros((n,n), dtype=int)-1
        if n%2 == 0:
            for i in range(0,n-2,2):
                for j in range(0,n-2,2):
                    self.kind[i:n:2, j:n:2] = 0
                    self.kind[i:n:2, j+1:n:2] = 1
                    self.kind[i+1:n:2, j:n:2] = 2
                    self.kind[i+1:n:2, j+1:n:2] = 3
        else:
            for i in range(0,n-2,2):
                for j in range(0,n-2,2):
                    if b.row % 2 == 0 and b.column % 2 == 0:
                        self.kind[i:n:2, j:n:2] = 0
                        self.kind[i:n:2, j+1:n:2] = 1
                        self.kind[i+1:n:2, j:n:2] = 2
                        self.kind[i+1:n:2, j+1:n:2] = 3
                    elif b.row % 2 == 0 and b.column % 2 == 1:
                        self.kind[i:n:2, j:n:2] = 1
                        self.kind[i:n:2, j+1:n:2] = 0
                        self.kind[i+1:n:2, j:n:2] = 3
                        self.kind[i+1:n:2, j+1:n:2] = 2
                    elif b.row % 2 == 1 and b.column % 2 == 0:
                        self.kind[i:n:2, j:n:2] = 2
                        self.kind[i:n:2, j+1:n:2] = 3
                        self.kind[i+1:n:2, j:n:2] = 0
                        self.kind[i+1:n:2, j+1:n:2] = 1
                    elif b.row % 2 == 1 and b.column % 2 == 1:
                        self.kind[i:n:2, j:n:2] = 3
                        self.kind[i:n:2, j+1:n:2] = 2
                        self.kind[i+1:n:2, j:n:2] = 1
                        self.kind[i+1:n:2, j+1:n:2] = 0
            
        self.kind = np.ravel(self.kind)

    def __iter__(self):
        for i in range(len(self.ra_min)):
            row, col = i//self.n, i%self.n
            yield Brick(self.ra_min[i], self.ra_max[i],
                        self.dec_min[i], self.dec_max[i],
                        self.kind[i], row, col
                        )

#-------------------------------------------------------------------------
#- Plotting utility functions
def plotbrick(brick):
    import matplotlib.pyplot as plt
    color = 'rgyb'[brick.kind%4]
    x = [brick.ra_min, brick.ra_max, brick.ra_max, brick.ra_min]
    y = [brick.dec_min, brick.dec_min, brick.dec_max, brick.dec_max]
    plt.fill(x, y, color=color)

def outlinebrick(brick, polar=False):
    import matplotlib.pyplot as plt
    if polar:
        n = 20
        x = np.linspace(brick.ra_min, brick.ra_max, n) * np.pi / 180
        ra = np.concatenate([x, x[-1::-1]])
        dec = np.concatenate([np.ones(n)*brick.dec_min, np.ones(n)*brick.dec_max])
        if brick.dec_min < 0:
            dec = dec + 90
        else:
            dec = 90 - dec
        plt.polar(ra, dec, 'k:', lw=2)
    else:
        x = [brick.ra_min, brick.ra_max, brick.ra_max, brick.ra_min, brick.ra_min]
        y = [brick.dec_min, brick.dec_min, brick.dec_max, brick.dec_max, brick.dec_min]
        plt.plot(x, y, 'k:', lw=2)
    
def polarbrick(brick):
    import matplotlib.pyplot as plt
    color = 'rgyb'[brick.kind%4]
    ax = plt.axes(polar=True)
    n = 20
    x = np.linspace(brick.ra_min, brick.ra_max, n) * np.pi / 180
    ra = np.concatenate([x, x[-1::-1]])
    dec = np.concatenate([np.ones(n)*brick.dec_min, np.ones(n)*brick.dec_max])
    if brick.dec_min < 0:
        dec = dec + 90
    else:
        dec = 90 - dec
    plt.fill(ra, dec, axes=ax, color=color)
    
if __name__ == '__main__':
    #- Example code for writing brick+bricklets file
    from astropy.io import fits
    from astropy.table import Table
    bricksize = 1.0
    n = 4
    bricks = Bricks(1.0)
    
    #- Bricklet arrays to fill; collect these as lists of arrays and convert
    #- to a numpy array with np.concatenate at the end
    ra_min = list()
    ra_max = list()
    dec_min = list()
    dec_max = list()
    kind = list()

    for b in bricks:
        bx = Bricklets(n, b)
        ra_min.append(bx.ra_min)
        ra_max.append(bx.ra_max)
        dec_min.append(bx.dec_min)
        dec_max.append(bx.dec_max)
        kind.append(bx.kind)

    ra_min = np.concatenate(ra_min).astype(np.float32)
    ra_max = np.concatenate(ra_max).astype(np.float32)
    dec_min = np.concatenate(dec_min)
    dec_max = np.concatenate(dec_max)
    kind = np.concatenate(kind)

    #- Output results
    outfile = 'bricks-{:.2f}-{}.fits'.format(bricksize, n)
    brickdata = dict(
        ra_min = bricks.ra_min, ra_max=bricks.ra_max,
        dec_min = bricks.dec_min, dec_max = bricks.dec_max,
        kind = bricks.kind
        )
        
    brickletdata = dict(
        ra_min = ra_min, ra_max = ra_max,
        dec_min = dec_min, dec_max = dec_max,
        kind = kind
        )
        
    #- ensure ordering of columns
    columns = ['ra_min', 'ra_max', 'dec_min', 'dec_max', 'kind']
    brickdata = Table(brickdata, names=columns).as_array()
    brickletdata = Table(brickletdata, names=columns).as_array()

    hdus = fits.HDUList()
    hdus.append(fits.PrimaryHDU(None))
    hdus.append(fits.BinTableHDU(brickdata, name='BRICKS'))
    hdus.append(fits.BinTableHDU(brickletdata, name='BRICKLETS'))
    hdus.writeto(outfile, clobber=True)
    
    

#-------------------------------------------------------------------------
#- Scratch code for cutting and pasting
"""
reload(skybrick)
b = skybrick.Bricks(1.0)
ii = (0 <= b.ra_min) & (b.ra_max < 5) & (30 <= b.dec_min) & (b.dec_max < 35)
bx = skybrick.Bricks(1.0)[ii]
fig = plt.figure(figsize=(5,5))
fig.subplots_adjust(left=0.15, right=0.95, bottom=0.13, top=0.93)
plt.clf()
n = 5
%cpaste
for x in bx:
    bricklet = skybrick.Bricklets(n, x)
    for y in bricklet:
        skybrick.plotbrick(y)
    skybrick.outlinebrick(x)
plt.xlabel('RA [deg]')
plt.ylabel('dec [deg]')
plt.title('1 deg bricks -> 1/{} deg bricklets'.format(n))
plt.savefig('bricklet-{}.png'.format(n), dpi=72)
--

plt.figure()
[skybrick.polarbrick(x) for x in b[0:30]]

plt.figure()
for x in b[0:30]:
    [skybrick.polarbrick(y) for y in skybrick.Bricklets(5, x)]
    skybrick.outlinebrick(x, polar=True)

"""
