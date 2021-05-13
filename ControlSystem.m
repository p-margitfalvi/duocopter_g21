classdef ControlSystem < handle
    properties
        gains;
        waypoints;
        cur_waypoint_idx;
        y_prev;
        t_prev;
        firstIter;
        
        logs;
    end
    methods
        
        function obj = ControlSystem(gains, waypoints)
            obj.gains = gains;
            obj.waypoints = waypoints;
            
            obj.cur_waypoint_idx = 1;
            obj.y_prev = -1;
            obj.t_prev = -1;
            
            obj.firstIter = true;
        end
        
        function cur_waypoint_idx = updateCurWpt(obj, t)
            if obj.waypoints.Time(obj.cur_waypoint_idx) <= t
                cur_waypoint_idx = min(obj.cur_waypoint_idx + 1, numel(obj.waypoints.Time));
                obj.cur_waypoint_idx = cur_waypoint_idx;
            else
                cur_waypoint_idx = obj.cur_waypoint_idx;
            end
        end
        
        function thrust = calculate(obj, y, t, v)
            wpt = obj.updateCurWpt(t);
            
            v_req = (obj.waypoints.Data(wpt) - y) / (obj.waypoints.Time(wpt) - t);
            
            if ~obj.firstIter
                obj.firstIter = false;
                v_cur = (y - obj.y_prev)/(t - obj.t_prev); % TODO: Use more points for smoother derivative
            else
                v_cur = 0;
            end
            
            v_cur = v;
            
            obj.y_prev = y;
            obj.t_prev = t;
            
            err = v_req - v_cur;
            
            thrust = obj.gains(1) * err;
            
            thrust = max(min(thrust, 1), 0);
        end
    end
end