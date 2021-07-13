{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "abroad-swift",
   "metadata": {},
   "source": [
    "# Hopfield Network"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "corporate-tablet",
   "metadata": {},
   "source": [
    "Developer: Thiago Fellipe Ortiz de Camargo"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "median-contest",
   "metadata": {},
   "source": [
    "## Standard Imports"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "synthetic-cleaning",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "lucky-contamination",
   "metadata": {},
   "source": [
    "## Hopfield"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 361,
   "id": "charitable-blade",
   "metadata": {},
   "outputs": [],
   "source": [
    "class hopfield(object):\n",
    "    \n",
    "    def __init__(self, patterns, noise_percentage, pattern_n_row, pattern_n_column, ib, epochs):\n",
    "        self.patterns = patterns\n",
    "        self.noise    = 1-noise_percentage\n",
    "        self.nrow     = pattern_n_row\n",
    "        self.ncol     = pattern_n_column\n",
    "        self.fmn      = len(patterns)\n",
    "        self.dim      = len(self.patterns[0])\n",
    "        self.ib       = ib\n",
    "        self.epc      = epochs\n",
    "        self.scape    = False\n",
    "        \n",
    "    def noise_attribution(self, patt):\n",
    "        self.pattern = patt\n",
    "        self.randM   = np.random.rand(self.nrow,self.ncol)\n",
    "        self.auxA    = self.noise > self.randM\n",
    "        self.auxB    = self.noise < self.randM\n",
    "        self.randM[self.auxA] =  1\n",
    "        self.randM[self.auxB] = -1\n",
    "        self.new_patter       = self.pattern.reshape(self.nrow,self.ncol)*self.randM\n",
    "        return self.new_patter.reshape(self.dim,1)\n",
    "    \n",
    "    def weights(self):\n",
    "        self.auxW = 0\n",
    "        \n",
    "        for patt in self.patterns:\n",
    "            self.auxW += patt*patt.reshape(self.dim,1)\n",
    "            \n",
    "        self.W = ((1/self.dim)*self.auxW)-((self.fmn/self.dim)*np.zeros((self.dim,self.dim)))\n",
    "        \n",
    "    \n",
    "    def run(self):\n",
    "        \n",
    "        self.outputs    = pd.DataFrame()\n",
    "        self.noised_img = pd.DataFrame()\n",
    "        for patt, i in zip(self.patterns,range(self.fmn)):\n",
    "            self.weights()\n",
    "            self.v_current  = self.noise_attribution(patt)\n",
    "            self.noised_img = pd.concat((self.noised_img, pd.DataFrame(self.v_current).T))\n",
    "            self.it = 0\n",
    "            self.scape = False\n",
    "\n",
    "            while(self.scape == False):\n",
    "                self.v_past    = self.v_current\n",
    "                self.u         = np.dot(self.W,self.v_past)+self.ib\n",
    "                self.v_current = np.sign(np.tanh(self.u))\n",
    "\n",
    "                if pd.DataFrame(self.v_current).equals(pd.DataFrame(self.v_past)):\n",
    "                    self.scape = True\n",
    "\n",
    "                if(self.it >= self.epc):\n",
    "                    self.scape = True\n",
    "\n",
    "                self.it += 1\n",
    "                \n",
    "            self.outputs = pd.concat((self.outputs,pd.DataFrame(self.v_current).T))\n",
    "    \n",
    "    "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "static-december",
   "metadata": {},
   "source": [
    "## Examples"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 332,
   "id": "static-roommate",
   "metadata": {},
   "outputs": [],
   "source": [
    "N1 = np.array((-1, -1, 1, 1, -1, -1, 1, 1, 1, -1, -1, -1, 1, 1, -1, -1, -1, 1, 1, -1, -1, -1, 1, 1, -1, -1, -1, 1, 1, -1, -1, -1, 1, 1, -1, -1, -1, 1, 1, -1, -1, -1, 1, 1, -1))\n",
    "N2 = np.array((1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1))\n",
    "N3 = np.array((1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1))\n",
    "N4 = np.array((1, 1, -1, 1, 1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, 1, 1, -1, -1, -1, 1, 1, -1, -1, -1, 1, 1, -1, -1, -1, 1, 1))\n",
    "N = np.array((N1, N2, N3, N4))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 333,
   "id": "altered-result",
   "metadata": {},
   "outputs": [],
   "source": [
    "hp = hopfield(patterns=N, noise_percentage=0.15, \n",
    "              pattern_n_row=9, pattern_n_column=5, ib=0, epochs=1000)\n",
    "hp.run()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 334,
   "id": "hairy-silly",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 504x720 with 12 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "\n",
    "fig, axs = plt.subplots(nrows=3, ncols=4, figsize=(7, 10))\n",
    "\n",
    "# ------- N1 -------\n",
    "axs[0][0].set_title('Pattern 1')\n",
    "axs[0][0].imshow(N1.reshape(9,5))\n",
    "\n",
    "axs[1][0].set_title('Noised Pattern 1')\n",
    "axs[1][0].imshow(hp.noised_img.iloc[0,:].values.reshape(9,5))\n",
    "\n",
    "axs[2][0].set_title('HP Pattern 1')\n",
    "axs[2][0].imshow(hp.outputs.iloc[0,:].values.reshape(9,5))\n",
    "\n",
    "\n",
    "# ------- N2 -------\n",
    "axs[0][1].set_title('Pattern 2')\n",
    "axs[0][1].imshow(N2.reshape(9,5))\n",
    "\n",
    "axs[1][1].set_title('Noised Pattern 2')\n",
    "axs[1][1].imshow(hp.noised_img.iloc[1,:].values.reshape(9,5))\n",
    "\n",
    "axs[2][1].set_title('HP Pattern 2')\n",
    "axs[2][1].imshow(hp.outputs.iloc[1,:].values.reshape(9,5))\n",
    "\n",
    "\n",
    "# ------- N3 -------\n",
    "axs[0][2].set_title('Pattern 3')\n",
    "axs[0][2].imshow(N3.reshape(9,5))\n",
    "\n",
    "axs[1][2].set_title('Noised Pattern 3')\n",
    "axs[1][2].imshow(hp.noised_img.iloc[2,:].values.reshape(9,5))\n",
    "\n",
    "axs[2][2].set_title('HP Pattern 3')\n",
    "axs[2][2].imshow(hp.outputs.iloc[2,:].values.reshape(9,5))\n",
    "\n",
    "\n",
    "# ------- N4 -------\n",
    "axs[0][3].set_title('Pattern 4')\n",
    "axs[0][3].imshow(N4.reshape(9,5))\n",
    "\n",
    "axs[1][3].set_title('Noised Pattern 4')\n",
    "axs[1][3].imshow(hp.noised_img.iloc[3,:].values.reshape(9,5))\n",
    "\n",
    "axs[2][3].set_title('HP Pattern 4')\n",
    "axs[2][3].imshow(hp.outputs.iloc[3,:].values.reshape(9,5))\n",
    "\n",
    "\n",
    "\n",
    "plt.show()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
