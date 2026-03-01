pre allocated buffers in byte tracker, currently arrays are dynamic using concat
instead of concat, we can design around a max_tracker_buffer_len constraint 
would allow for no dynamic access needed