// ./cvv_text_det nature.png nature1.png

extern "C" {
#include "ccv.h"
}

#include <sys/time.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <tesseract/baseapi.h>

#include <visp_naoqi/vpNaoqiGrabber.h>

#include <alproxies/altexttospeechproxy.h>


static unsigned int get_current_time(void)
{
    struct timeval tv;
    gettimeofday(&tv, 0);
    return tv.tv_sec * 1000 + tv.tv_usec / 1000;
}

using namespace cv;

int main(int argc, const char* argv[]){


    std::string opt_ip = "198.18.0.1";

    if (argc == 3) {
        if (std::string(argv[1]) == "-ip")
            opt_ip = argv[2];
    }


    // Open Proxy for the speech
    AL::ALTextToSpeechProxy tts(opt_ip, 9559);
    tts.setLanguage("English");

    vpNaoqiGrabber g;
    if (! opt_ip.empty()) {
        std::cout << "Connect to robot with ip address: " << opt_ip << std::endl;
        g.setRobotIp(opt_ip);
    }

    g.setCameraResolution(AL::k4VGA);
    g.open();
    g.setCamera(2);
    g.setFramerate(2);

    std::cout << "Image size: " << g.getWidth() << " " << g.getHeight() << std::endl;
    // Create an OpenCV image container
    cv::Mat img_ocv = cv::Mat(cv::Size(g.getWidth(), g.getHeight()), CV_8UC3);

    g.acquire(img_ocv);

    Mat gray_image;
    cvtColor( img_ocv, gray_image, CV_BGR2GRAY );

    //imshow("gray_image",gray_image);
   // waitKey( 0 );

    Mat gray_image_one; //= cv::Mat(cv::Size(g.getWidth(), g.getHeight()), CV_8UC1);;
    // gray_image.convertTo(gray_image_one, CV_8UC1, 255.0/2048.0);

    // imshow("gray_image_one",gray_image_one);
    //    waitKey( 0 );

    vector<Mat> split_image(3);
    split(gray_image, split_image);

    merge(split_image,gray_image_one);

    Mat convertedTo8UC1;
    gray_image_one.convertTo(convertedTo8UC1, CV_8UC1);

   // imshow("convertedTo8UC1",convertedTo8UC1);
  //  waitKey( 0 );

    //    temp.convertTo(gray_image_one, CV_8UC1);
    //    imshow("gray_image_one",gray_image_one);
    //    waitKey( 0 );

    //    Mat img_ocv;
    //    img_ocv = imread("TestImage1.jpg",CV_LOAD_IMAGE_GRAYSCALE | CCV_IO_NO_COPY);
    ccv_dense_matrix_t* image = 0;
    ccv_read(convertedTo8UC1.data, &image, CCV_IO_GRAY_RAW, convertedTo8UC1.rows, convertedTo8UC1.cols, convertedTo8UC1.step[0]);
    // ccv_read("1.jpg", &image, CCV_IO_GRAY | CCV_IO_ANY_FILE);

    // std::vector<Rect> roi;

    Mat img_disp = img_ocv.clone();

    // Tesseract
    tesseract::TessBaseAPI tess;
    if( tess.Init(NULL, "eng", tesseract::OEM_DEFAULT))
    {
        fprintf(stderr, "Could not initialize tesseract.\n");
        exit(1);
    }

    //tess.SetPageSegMode(tesseract::PSM_SINGLE_BLOCK);
    tess.SetPageSegMode(tesseract::PSM_SINGLE_WORD);


//    tess.SetVariable("tessedit_char_whitelist",
//                     "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
//                    "0123456789");
//    tess.SetVariable("tessedit_char_whitelist",
//                     "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ");



    tess.SetVariable("tessedit_char_whitelist", "0123456789");
   tess.SetVariable("language_model_penalty_non_dict_word", "0");





    unsigned int elapsed_time = get_current_time();
    ccv_array_t* words = ccv_swt_detect_words(image, ccv_swt_default_params);


    elapsed_time = get_current_time() - elapsed_time;
    if (words)
    {

        for (unsigned int i = 0; i < words->rnum; i++)
        {
            ccv_rect_t* rect = (ccv_rect_t*)ccv_array_get(words, i);
            printf("%d %d %d %d\n", rect->x, rect->y, rect->width, rect->height);

            //cv::Rect roi_rect(rect->x-5, rect->y-5,  rect->width+10, rect->height+10);


            int x = rect->x-2;
            int y = rect->y-2;
            int w = rect->width+4;
            int h = rect->height+4;

            if (x < 0)
                x = 0;
            if (y < 0)
                y = 0;
            if (w > g.getWidth())
                w = g.getWidth();
            if (h > g.getHeight())
                w = g.getHeight();

            cv::Rect roi_rect(x, y,  w, h);


            //roi.push_back(roi_rect);

            if ( (rect->width * rect->height) >= 800 )
            {

                //x, y, x + width, y + height)
                rectangle( img_disp, Point( rect->x, rect->y ), Point( rect->x + rect->width , rect->y + rect->height ), Scalar(255, 0, 0, 0),2, 8, 0 );

                Mat image_roi = gray_image(roi_rect);
                Mat binary_roi;

                GaussianBlur( image_roi, image_roi, Size( 5, 5 ), 0, 0 );
                threshold(image_roi,binary_roi, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);

                //imshow("Image cropped",image_roi);
                //imwrite("cropped.jpg"+i, image_roi);
                //imshow("Image cropped bn", binary_roi);
                //imwrite("cropped_bn.jpg"+i, binary_roi);



                Mat binary_roi_resized;
                resize(binary_roi, binary_roi_resized, Size(), 4.0, 4.0);



                //imshow("final", binary_roi_resized);

                tess.SetImage((uchar*)binary_roi_resized.data, binary_roi_resized.cols, binary_roi_resized.rows, 1, binary_roi_resized.cols);

                // Get the text
                char* out = tess.GetUTF8Text();
                std::cout << "Result: " << out << std::endl;


                std::vector<int> compression_params; //vector that stores the compression parameters of the image
                compression_params.push_back(CV_IMWRITE_JPEG_QUALITY); //specify the compression technique
                compression_params.push_back(100); //specify the compression quality
                std::string nameFile = std::string(out) + ".jpg";
                bool bSuccess = cv::imwrite(nameFile, binary_roi_resized, compression_params); //write the image to file



                int id = tts.post.say(out);
                //tts.wait(id,500);
                //waitKey( 0 );

            }


        }
        printf("total : %d in time %dms\n", words->rnum, elapsed_time);
        ccv_array_free(words);
    }
    imshow("Image",img_disp);
    std::vector<int> compression_params; //vector that stores the compression parameters of the image
    compression_params.push_back(CV_IMWRITE_JPEG_QUALITY); //specify the compression technique
    compression_params.push_back(100); //specify the compression quality
     cv::imwrite("Result.jpg", img_disp, compression_params); //write the image to file
     cv::imwrite("Original.jpg", img_ocv, compression_params); //write the image to file




    //    Mat image_roi = img_ocv(roi[0]);
    //    Mat binary_roi;
    //    threshold(image_roi,binary_roi, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
    //    imshow("Image cropped",image_roi);
    //    imwrite("cropped.jpg", image_roi);
    //    imshow("Image cropped bn",binary_roi);
    //    imwrite("cropped_bn.jpg", binary_roi);

    //    // Pass it to Tesseract API
    //    tesseract::TessBaseAPI tess;
    //    tess.Init(NULL, "fra", tesseract::OEM_DEFAULT);
    //    tess.SetPageSegMode(tesseract::PSM_SINGLE_BLOCK);
    //    tess.SetImage((uchar*)binary_roi.data, binary_roi.cols, binary_roi.rows, 1, binary_roi.cols);

    //    // Get the text
    //    char* out = tess.GetUTF8Text();
    //    std::cout << "Result: " << out << std::endl;

    waitKey( 0 );
    ccv_matrix_free(image);

    ccv_drain_cache();

    return 0;
}
