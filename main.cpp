#include <arrayfire.h>
#include <iostream>
#include <cstdio>
#include <cstdlib>

using namespace af;
using namespace std;

//static const int samples = 1e7;
static const int SAMPLES = 1000;
static int WIDTH=1000, HEIGHT=400;

static float boundary_coors[] = { 0,     0,
                                  WIDTH, 0,
                                  WIDTH, HEIGHT,
                                  0,     HEIGHT };

static float boundary_vec_const[] =  { WIDTH,  0,
                                  0, HEIGHT,
                                  -WIDTH,  0,
                                  0, -HEIGHT };

float covariance_consts[] = { 0.2, af::Pi/16 }; //variance on state update for displacement and heading
//TODO: variance? or variance^^ ?

void state_update(array &state_vecs, float dx, float dth, array covariance);
af::array sensor_distances(af::array &state_vecs);
af::array resample(int iterations, af::array weights, af::array indices);

int main(int argc, char *argv[]) {
    try {
        // Select a device and display arrayfire info
        int device = argc > 1 ? atoi(argv[1]) : 0;
        af::setDevice(device);
        af::info();
        //initialize arrayfire window
        af::Window myWindow(WIDTH, HEIGHT, "Particle Filtering Example: ArrayFire");

        //create a set of random particles
        array particles = randu(SAMPLES, 3); // <x,y,theta>
        particles.col(0) *= WIDTH;
        particles.col(1) *= HEIGHT;
        particles.col(2) *= 2.f * af::Pi;

        //set initial speed and orientation of particles
        float v = 10;           // pixels/step
        float w  = -af::Pi/2;   // rad/step
        particles(0,0) = WIDTH/2; particles(0,1) = HEIGHT/2;

        array weights = constant(1.f/SAMPLES, SAMPLES); //set initial weights to equal 1/N

        timer timestep = timer::start();
        timer runtime  = timer::start();
        while(!myWindow.close()) {
            float t  = timer::stop(runtime);
            float dt = timer::stop(timestep);
            timestep = timer::start();
            setSeed(time(NULL));

            //myWindow.grid(2, 1);
            array c(2, 1, &covariance_consts[0]);
            array noise_cov = diag(c, 0, false);

            //particles.cols(0,1) += noise;
            v += sin(t); w = af::Pi/3 * sin(t/4);
            state_update(particles, v * dt, w * dt, noise_cov);

            array distance_vector = sensor_distances(particles);
            float actual_distance = distance_vector.scalar<float>();
            //assign weights to each particle
            weights = exp(-0.5 * (distance_vector - constant(actual_distance, SAMPLES) * 0.1)) + 0.01;
            //normalize weights
            float wsum = af::sum(weights).scalar<float>();
            weights /= wsum;

            array sorted, indices;
            sort(sorted, indices, weights);
            array sample_space = accum(weights);

            array resampled_indices = resample(5, weights, indices);
            array xs = particles.col(0);
            array ys = particles.col(1);
            array ts = particles.col(2);
            af_print(xs);
            af_print(particles.col(0));
            particles.col(0) = xs(resampled_indices);
            particles.col(1) = ys(resampled_indices);
            particles.col(2) = ts(resampled_indices);
            //af_print(particles);

            array image = constant(0, HEIGHT, WIDTH, f32);
            array ids(SAMPLES, u32);
            ids = (particles.col(0).as(u32) * (HEIGHT)) + particles.col(1).as(u32);
            image(ids) = 0.5;
            image((particles(0,0).as(u32) * (HEIGHT)) + particles(0,1).as(u32)) = 1.f;

            array mask = constant(1, 3, 3);
            image = dilate(image, mask);

            myWindow.image(image);
            //myWindow(0,0).image(image);
            //myWindow(1,0).plot(particles.col(0), particles.col(1));
            //myWindow.show();
        }
    } catch (af::exception& e) {
        fprintf(stderr, "%s\n", e.what());
        throw;
    }

    return 0;
}


void state_update(af::array &state_vecs, float dx, float dth, array covariance){
    //get noise for each particle
    array noise = matmul(randn(SAMPLES, 2), covariance);

    //update state vector for each particle while taking noise into account
    state_vecs.col(0) += (dx+noise.col(0)) * cos(state_vecs.col(2) + dth + noise.col(1));
    state_vecs.col(1) += (dx+noise.col(0)) * sin(state_vecs.col(2) + dth + noise.col(1));
    state_vecs.col(2) += dth + noise.col(1);
}

af::array sensor_distances(af::array &state_vecs){
    //TODO symmetric distance testing
    array boundaries(4, 2, boundary_coors);
    array sensor_endpoints(SAMPLES, 2);
    sensor_endpoints.col(0) = 100/cos(state_vecs.col(0));
    sensor_endpoints.col(1) = 100/sin(state_vecs.col(1));

    array boundary_vecs(4, 2, boundary_vec_const);
    array sensor_vectors = sensor_endpoints - state_vecs.cols(0,1);

    //incorrect shift?
    array perp_svecs = shift(sensor_vectors, 1);
    perp_svecs.col(1) = -1 * perp_svecs.col(1);
    array perp_bvecs = shift(boundary_vecs,  1);
    perp_bvecs.col(1) = -1 * perp_bvecs.col(1);

    array vec_diff(SAMPLES, 4);
    vec_diff.col(0)  = diag(matmul((state_vecs.cols(0,1) - tile(boundaries.row(0), SAMPLES, 1)), perp_svecs, AF_MAT_NONE, AF_MAT_TRANS))
        / diag(matmul(tile(boundaries.row(0), SAMPLES, 1), perp_svecs, AF_MAT_NONE, AF_MAT_TRANS)) ;
    vec_diff.col(1)  = diag(matmul((state_vecs.cols(0,1) - tile(boundaries.row(1), SAMPLES, 1)), perp_svecs, AF_MAT_NONE, AF_MAT_TRANS))
        / diag(matmul(tile(boundaries.row(1), SAMPLES, 1), perp_svecs, AF_MAT_NONE, AF_MAT_TRANS)) ;
    vec_diff.col(2)  = diag(matmul((state_vecs.cols(0,1) - tile(boundaries.row(2), SAMPLES, 1)), perp_svecs, AF_MAT_NONE, AF_MAT_TRANS))
        / diag(matmul(tile(boundaries.row(2), SAMPLES, 1), perp_svecs, AF_MAT_NONE, AF_MAT_TRANS)) ;
    vec_diff.col(3)  = diag(matmul((state_vecs.cols(0,1) - tile(boundaries.row(3), SAMPLES, 1)), perp_svecs, AF_MAT_NONE, AF_MAT_TRANS))
        / diag(matmul(tile(boundaries.row(3), SAMPLES, 1), perp_svecs, AF_MAT_NONE, AF_MAT_TRANS)) ;

    vec_diff(vec_diff < 0 ) = 9;
    vec_diff(vec_diff > 1 ) = 9;

    return min(vec_diff,1);
}


af::array resample(int iterations, af::array weights, af::array indices) {
    array resampled = indices;
    for(int i=0;i<iterations;i++){
        array u = randu(SAMPLES); // <x,y,theta>
        array j = shift(indices, (SAMPLES+i)/2);

        array ratio = weights(indices) / weights(j);
        resampled(ratio < u) = shift(indices, (SAMPLES+i)/2)(ratio < u);
    }

    //af_print(resampled);
    resampled(0) = 0;
    return resampled;
}
