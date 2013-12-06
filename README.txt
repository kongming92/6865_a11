Name: Charles Liu
MIT Email: cliu2014@mit.edu

Q1:How long did the assignment take?:
{A1: about 18 hours}

Q2:Potential issues with your solution and explanation of partial completion (for partial credit):
{A2: It seems everything is correct. For the optimization, not sure how much room there is to improve though.}

Q3:Anything extra you may have implemented:
{A3: none}

Q4:Collaboration acknowledgement (but again, you must write your own code):
{A4: Ryan Lacey}

Q5:What was most unclear/difficult?:
{A5: The Halide scheduling options, specifically what happens with compute_at}

Q6:What was most exciting?:
{A6: Tuning and watching the Harris corner speedup}

Q7: How long did it take for the 2 schedules for the smooth gradient?
{A7:
	Default 1.093ms (2.277ms/megapixel),
	Best 0.760ms (1.584ms/megapixel)
	Speedup factor: 1.438
}

Q8: Speed in ms per megapixel for the 4 schedules (1 per line)
{A8:
	Root 155.05ms for hk.png (4.34ms/megapixel)
	Inline 156.04ms (4.367ms/megapixel)
	Tiling 197.88ms (5.538ms/megapixel)
	Tile & parallel 60.989ms (1.707ms/megapixel)
}

Q9: What machine did you use (CPU type, speed, number of cores, memory)
{A9: Athena dialup: Intel Xeon X5460, 3.16GHz, 12M cache, 4 cores, 12 GB memory (I probably don't have access to all of the memory)}

Q10: Speed for the box schedules, and best tile size
{A10:
	Default 4964.6ms (138.85ms/mp)
	Root first stage 2513.3ms (70.296ms/mp)
	256x256 tile + interleave 1461.2ms (40.87ms/mp)
	256x256 tile + parallel 645.85ms (18.06ms/mp)
	256x256 tile + parallel vector no interleave 640.64ms (17.92ms/mp)

	Best tile size: 64x64, 541.19ms (15.13ms/mp)
}

Q11: How fast did Fredo’s Harris and your two schedules were on your machine?
{A11:
	Fredo's Harris: 23.6 seconds (660ms/mp)
	All root 12.65 seconds (353.8ms/mp)
	My best schedule 1.88 seconds (52.7ms/mp)
}

Q12: Describe your auto tuner in one paragraph
{A12:
	I identified a few places that are candidates for tiling. Specifically, the top-level function, the blur of the tensor components (ix, iy, ixiy), and the blur of the luminance. Then, I schedule the x-component of the blur to be the producer for the blur, so that blur_x.compute_at(blur, ...).The things that I try are the different tile sizes for each of these tiles (increasing powers of 2). The other thing that I try is to figure out where the producer should be computed. After playing with things, it appears that the tensor components ix, iy, and ixiy blurred (in both directions) perform best when using compute_root, while the x-direction blurred should be computed locally. Thus, for those, I test whether computing at xo or yo would be more effective.

	For hk.png, I kept getting out-of-memory errors while trying different compute_at values, so I just set it to compute_at xo, which I found was usually the best-performing option.

	I combined all of these results for the final Harris schedule.
}

Q13: What is the best schedule you found for Harris.
{A13: Tile the top-level function at 256x256. Tile the tensor calculations at 128x128. Tile the luminance blurs at 64x64. Compute everything else root, except the x-components of the blurs, which are scheduled as they are needed. Also compute lumi inline since it does not rely on other points. Finally, vectorize the inner loops on x, and unroll the inner loops on y (the vectorize makes things a lot faster, the unroll not so much)}