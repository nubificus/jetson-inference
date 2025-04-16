#define _POSIX_C_SOURCE 200809L
#define _DEFAULT_SOURCE
#include <linux/limits.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

/*
 * Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "videoSource.h"
#include "videoOutput.h"

#include "cudaFont.h"
#include "imageNet.h"

#include <signal.h>
#include <fcntl.h>

#include <chrono>

bool signal_recieved = false;

void sig_handler(int signo)
{
	if( signo == SIGINT )
	{
		LogVerbose("received SIGINT\n");
		signal_recieved = true;
	}
}

int fs_file_read(const char *path, void **data, size_t *size)
{
	if (!path || !data)
		return EINVAL;

	int fd = open(path, O_RDONLY);
	if (fd == -1) {
		fprintf(stderr,"Could not open file %s: %s", path,
			     strerror(errno));
		return errno;
	}

	/* Find the size of the file */
	struct stat stat;
	int ret = fstat(fd, &stat);
	if (ret < 0) {
		fprintf(stderr, "Could not fstat file %s: %s", path,
			     strerror(errno));
		ret = errno;
		return ret;
	}
	if (!stat.st_size) {
		fprintf(stderr, "File %s is empty", path);
		ret = EINVAL;
		return ret;
	}

	unsigned char *buff = (unsigned char*)malloc(stat.st_size);
	if (!buff) {
		fprintf(stderr,"Could not malloc buff: %s", strerror(errno));
		ret = errno;
		return ret;
	}

	size_t bytes = stat.st_size;
	ssize_t ptr = 0;
	while (bytes) {
		ssize_t rret = read(fd, &buff[ptr], bytes);
		if (rret < 0) {
			fprintf(stderr,"Could not read file %s: %s", path,
				     strerror(errno));
			free(buff);
			ret = errno;
			return ret;
		}

		ptr += rret;
		bytes -= rret;
	}

	*data = buff;
	if (size)
		*size = ptr;

	close(fd);

	if (ret)
		return ret;
	return 0;
}



int usage()
{
	printf("usage: imagenet [--help] [--network=NETWORK] ...\n");
	printf("                input_URI [output_URI]\n\n");
	printf("Classify a video/image stream using an image recognition DNN.\n");
	printf("See below for additional arguments that may not be shown above.\n\n");	
	printf("optional arguments:\n");
	printf("  --help            show this help message and exit\n");
	printf("  --network=NETWORK pre-trained model to load (see below for options)\n");
	printf("  --topK=N   	   show the topK number of class predictions (default: 1)\n");
	printf("positional arguments:\n");
	printf("    input_URI       resource URI of input stream  (see videoSource below)\n");
	printf("    output_URI      resource URI of output stream (see videoOutput below)\n\n");

	printf("%s", imageNet::Usage());
	printf("%s", videoSource::Usage());
	printf("%s", videoOutput::Usage());
	printf("%s", Log::Usage());

	return 0;
}
#include <stb/stb_image.h>
#include <stb/stb_image_write.h>
#include <jetson-utils/cudaMappedMemory.h>
#include <jetson-utils/filesystem.h>

bool loadImageBufRGBA(const void *buffer, int buf_len, float4 **cpu, float4 **gpu,
		int *width, int *height, const float4& mean)
{
	// validate parameters
	if (!buffer || !cpu || !gpu || !width || !height) {
		fprintf(stderr, "loadImageRGBA() - invalid parameter(s)\n");
		return false;
	}

	// attempt to load the data from disk
	int imgWidth = 0;
	int imgHeight = 0;
	int imgChannels = 0;

	unsigned char* img = stbi_load_from_memory((const unsigned char *)buffer,
			buf_len, &imgWidth, &imgHeight, &imgChannels, 0);
	if (!img) {
		fprintf(stderr, "Failed to load img from memory\n");
		return false;
	}

	if (*width > 0 && *height > 0) {
		// TODO: Resize
	}

	// allocate CUDA buffer for the image
	const size_t imgSize = imgWidth * imgHeight * sizeof(float) * 4;

	if (!cudaAllocMapped((void **)cpu, (void **)gpu, imgSize)) {
		fprintf(stderr, "Failed to allocate %zu bytes for image\n", imgSize);
		return false;
	}


	// convert uint8 image to float4
	float4 *cpuPtr = *cpu;
	for (int y = 0; y < imgHeight; y++) {
		const size_t yOffset = y * imgWidth * imgChannels * sizeof(unsigned char);

		for (int x = 0; x < imgWidth; x++) {
			#define GET_PIXEL(channel) float(img[offset + channel])
			#define SET_PIXEL_FLOAT4(r,g,b,a) cpuPtr[y*imgWidth+x] = make_float4(r,g,b,a)

			const size_t offset = yOffset + x * imgChannels * sizeof(unsigned char);

			switch (imgChannels) {
			case 1:
			{
				const float grey = GET_PIXEL(0);
				SET_PIXEL_FLOAT4(grey - mean.x, grey - mean.y, grey - mean.z, 255.0f - mean.w);
				break;
			}
			case 2:
			{
				const float grey = GET_PIXEL(0);
				SET_PIXEL_FLOAT4(grey - mean.x, grey - mean.y, grey - mean.z, GET_PIXEL(1) - mean.w);
				break;
			}
			case 3:
			{
				SET_PIXEL_FLOAT4(GET_PIXEL(0) - mean.x, GET_PIXEL(1) - mean.y, GET_PIXEL(2) - mean.z, 255.0f - mean.w);
				break;
			}
			case 4:
			{
				SET_PIXEL_FLOAT4(GET_PIXEL(0) - mean.x, GET_PIXEL(1) - mean.y, GET_PIXEL(2) - mean.z, GET_PIXEL(3) - mean.w);
				break;
			}
			}
		}
	}

	*width  = imgWidth;
	*height = imgHeight;

	free(img);
	return true;
}

// limit_pixel
static inline unsigned char limit_pixel( float pixel, float max_pixel )
{
	if( pixel < 0 )
		pixel = 0;

	if( pixel > max_pixel )
		pixel = max_pixel;

	return (unsigned char)pixel;
}

int main( int argc, char** argv )
{
	/*
	 * parse command line
	 */
	commandLine cmdLine(argc, argv);

	if( cmdLine.GetFlag("help") )
		return usage();


	/*
	 * attach signal handler
	 */
	if( signal(SIGINT, sig_handler) == SIG_ERR )
		LogError("can't catch SIGINT\n");


	/*
	 * create input stream
	 */
	videoSource* input = videoSource::Create(cmdLine, ARG_POSITION(0));

	if( !input )
	{
		LogError("imagenet:  failed to create input stream\n");
		return 1;
	}


	/*
	 * create output stream
	 */
	videoOutput* output = videoOutput::Create(cmdLine, ARG_POSITION(1));
	
	if( !output )
	{
		LogError("imagenet:  failed to create output stream\n");	
		return 1;
	}
	

	/*
	 * create font for image overlay
	 */
	cudaFont* font = cudaFont::Create();
	
	if( !font )
	{
		LogError("imagenet:  failed to load font for overlay\n");
		return 1;
	}


	/*
	 * create recognition network
	 */
	imageNet* net = imageNet::Create(cmdLine);
	
	if( !net )
	{
		LogError("imagenet:  failed to initialize imageNet\n");
		return 1;
	}

	const int topK = cmdLine.GetInt("topK", 1);  // by default, get only the top result
	
	std::chrono::time_point<std::chrono::high_resolution_clock> start, start_total;
	std::chrono::time_point<std::chrono::high_resolution_clock> end, end_total;
	std::chrono::duration<long, std::nano> duration, total;

	
	/*
	 * processing loop
	 */
	unsigned long sum = 0;
	long iters = 0;
	while( !signal_recieved )
	{
		// capture next image
		uchar3* image = NULL;
		int status = 0;
		
#if 1
		if( !input->Capture(&image, &status) )
		{
			if( status == videoSource::TIMEOUT )
				continue;
			
			break; // EOS
		}
#endif
	size_t len_img;

	int ret = fs_file_read("/usr/local/share/vaccel/images/example.jpg", (void **)&image, &len_img);

		/*
		 * load image from disk
		 */
		float4 *imgCPU = NULL;
		float4 *imgCUDA = NULL;
		int imgWidth  = 0;
		int imgHeight = 0;

		start = std::chrono::high_resolution_clock::now();
		if (!loadImageBufRGBA(image, len_img, (float4**)&imgCPU, (float4**)&imgCUDA, &imgWidth, &imgHeight, make_float4(0,0,0,0))) {
			fprintf(stderr, "imagenet: Failed to load image\n");
			return -1;
			//return VACCEL_ENOENT;
		}


		/*
		 * classify image
		 */
		float confidence = 0.0f;
#if 0
		const int img_class = net->Classify(imgCUDA, imgWidth, imgHeight, &confidence);
#endif


		// classify image - note that if you only want the top class, you can simply run this instead:
		// 	float confidence = 0.0f;
		//	const int img_class = net->Classify(image, input->GetWidth(), input->GetHeight(), &confidence);
		imageNet::Classifications classifications;	// std::vector<std::pair<uint32_t, float>>  (classID, confidence)

		
		
		//if( net->Classify(image, input->GetWidth(), input->GetHeight(), classifications, topK) < 0 )
		if( net->Classify(imgCUDA, imgWidth, imgHeight, classifications, topK) < 0 )
			continue;
		
		// draw predicted class labels
		for( uint32_t n=0; n < classifications.size(); n++ )
		{
			const uint32_t classID = classifications[n].first;
			const char* classLabel = net->GetClassLabel(classID);
			const float confidence = classifications[n].second * 100.0f;
			
			LogVerbose("imagenet:  %2.5f%% class #%i (%s)\n", confidence, classID, classLabel);	

#if 0
			char str[256];
			sprintf(str, "%05.2f%% %s", confidence, classLabel);
			font->OverlayText(image, input->GetWidth(), input->GetHeight(),
						   str, 5, 5 + n * (font->GetSize() + 5), 
						   make_float4(255,255,255,255), make_float4(0,0,0,100));
#endif
		}
		
		// render outputs
		if( output != NULL )
		{
			output->Render(image, input->GetWidth(), input->GetHeight());

			// update status bar
			char str[256];
			sprintf(str, "TensorRT %i.%i.%i | %s | Network %.0f FPS", NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR, NV_TENSORRT_PATCH, net->GetNetworkName(), net->GetNetworkFPS());
			output->SetStatus(str);	

			// check if the user quit
			if( !output->IsStreaming() )
				break;
		}

		end = std::chrono::high_resolution_clock::now();
		duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
		//LogVerbose("imagenet:start:%lld, end:%lld  total: %lld\n", start, end, duration.count());
		LogVerbose("imagenet:total: %lld\n", duration.count());
		sum += duration.count();
		// print out timing info
		net->PrintProfilerTimes();
		iters++;
	}
	LogVerbose("imagenet:total: %lld, avg: %2.5f\n", (double)sum/iters);
	
	
	/*
	 * destroy resources
	 */
	LogVerbose("imagenet:  shutting down...\n");
	
	SAFE_DELETE(input);
	SAFE_DELETE(output);
	SAFE_DELETE(net);
	
	LogVerbose("imagenet:  shutdown complete.\n");
	return 0;
}

