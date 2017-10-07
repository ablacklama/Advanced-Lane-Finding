import numpy as np


class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False ###

        # x values of the last n fits of the line
        self.recent_xfitted = [] ###

        #average x values of the fitted line over the last n iterations
        self.bestx = None ###

        #polynomial coefficients of last n iterations
        self.recent_fits = [] ###

        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None ###

        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])] ###

        #radius of curvature of the line in some units
        self.radius_of_curvature = None

        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')

        #x values for detected line pixels
        self.allx = None ###

        #y values for detected line pixels
        self.ally = None ###

        #number of frames undetected
        self.num_missed = 0 ###

        self.curve_sanity = True


    def update_fit(self, fit, allx, ally, n_lines):


        #if there are to few points to rely on or a line wasn't detected
        if (len(np.unique(ally)) < 280) or not self.curve_sanity:
            self.num_missed += 1
            if (self.num_missed > 20):
                self.detected = False


        #if everything was normal and a line was detected
        else:

            self.num_missed = 0
            self.detected = True
            self.current_fit = np.array(fit)
            self.allx = allx
            self.ally = ally

            self.recent_fits.append(fit)
            self.best_fit = np.average(self.recent_fits, axis=0)
            if(len(self.recent_fits) > n_lines):
                self.recent_fits = self.recent_fits[-n_lines:]


            lowestx = allx[np.argmax(ally)]

            #keep n_lines amount of previus fits and use them to help compute the best fit
            self.recent_xfitted.append(lowestx)
            if (len(self.recent_xfitted) > n_lines):
                self.recent_xfitted = self.recent_xfitted[-n_lines:]
            self.bestx = np.average(self.recent_xfitted)






    def get_radius(self, r_line, polyfit_left, polyfit_right):
        self.curve_sanity = True
        r_line.curve_sanity = True
        if(type(polyfit_left) != type(None) and type(polyfit_right) != type(None)):

            ploty = np.linspace(0, 719, num=720)  # to cover same y-range as image
            y_eval = np.max(ploty)
            # Define conversions in x and y from pixels space to meters

            ym_per_pix = 30 / 720  # meters per pixel in y dimension
            xm_per_pix = 3.7 / 600  # meters per pixel in x dimension

            leftx = polyfit_left[0] * ploty ** 2 + polyfit_left[1] * ploty + polyfit_left[2]
            rightx = polyfit_right[0] * ploty ** 2 + polyfit_right[1] * ploty + polyfit_right[2]

            # Fit new polynomials to x,y in world space
            left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
            right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)
            # Calculate the new radii of curvature
            left_curverad = ((1 + (
            2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit_cr[0])
            right_curverad = ((1 + (
            2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
                2 * right_fit_cr[0])

            middle = 640
            l_base_pos = 720 * polyfit_left[0] ** 2 + 720 * polyfit_left[1] + polyfit_left[2]
            r_base_pos = 720 * polyfit_right[0] ** 2 + 720 * polyfit_right[1] + polyfit_right[2]
            lane_center = (r_base_pos + l_base_pos)/2
            offset = (middle - lane_center) * xm_per_pix

            if(left_curverad / right_curverad) > 2 or (left_curverad / right_curverad) < 0.5:
                self.curve_sanity = False
                r_line.curve_sanity = False

            return (left_curverad, right_curverad, offset)
        return (0,0)