#include <arrayfire.h>
#include <iostream>
#include <cstdio>
#include <cstdlib>

using namespace af;
using namespace std;

//static const int samples = 1e7;
static const int samples = 100;
static int WIDTH=800, HEIGHT=600;

float covariance_consts[] = { 1, af::Pi/16 }; //variance on state update for displacement and heading
//TODO: variance? or variance^^ ?

void state_update(array &state_vecs, float dx, float dth, array covariance);

int main(int argc, char *argv[]) {
    try {
        // Select a device and display arrayfire info
        int device = argc > 1 ? atoi(argv[1]) : 0;
        af::setDevice(device);
        af::info();
        //initialize arrayfire window
        af::Window myWindow(WIDTH, HEIGHT, "Particle Filtering Example: ArrayFire");

        //create a set of random particles
        array particles = randu(samples, 3); // <x,y,theta>
        particles.col(0) *= WIDTH;
        particles.col(1) *= HEIGHT;
        particles.col(2) *= 2.f * af::Pi;

        //set initial speed and orientation of particles
        float v = 10;   // pixels/step
        float w  = -af::Pi/2;   // rad/step
        particles(0,0) = WIDTH/2; particles(0,1) = HEIGHT/2;

        array weights = constant(1.f/samples, samples); //set initial weights to equal 1/N 

        timer timestep = timer::start();
        timer runtime  = timer::start();
        while(!myWindow.close()) {
            float t = timer::stop(runtime);
            float dt = timer::stop(timestep);
            timestep = timer::start();
            setSeed(time(NULL));

            //myWindow.grid(2, 1);
            array c(2, 1, &covariance_consts[0]);
            array noise_cov = diag(c, 0, false);

            //particles.cols(0,1) += noise;
            v += sin(t); w = af::Pi/3 * sin(t/4);
            state_update(particles, v * dt, w * dt, noise_cov);

            array image = constant(0, HEIGHT, WIDTH, f32);
            array ids(samples, u32);
            ids = (particles.col(0).as(u32) * (HEIGHT)) + particles.col(1).as(u32);
            image(ids) = 0.5;
            image((particles(0,0).as(u32) * (HEIGHT)) + particles(0,1).as(u32)) = 1.f;

            array mask = constant(1, 3, 3);
            image = dilate(image, mask);

            myWindow.image(image);
            //myWindow(0,0).image(image);
            //myWindow(1,0).plot(particles.col(0), particles.col(1));
            myWindow.show();
        }
    } catch (af::exception& e) {
        fprintf(stderr, "%s\n", e.what());
        throw;
    }

    return 0;
}


void state_update(af::array &state_vecs, float dx, float dth, array covariance){
    //get noise for each particle
    array noise = matmul(randn(samples, 2), covariance);

    //update state vector for each particle while taking noise into account
    state_vecs.col(0) += (dx+noise.col(0)) * cos(state_vecs.col(2) + dth + noise.col(1));
    state_vecs.col(1) += (dx+noise.col(0)) * sin(state_vecs.col(2) + dth + noise.col(1));
    state_vecs.col(2) += dth + noise.col(1);
}


