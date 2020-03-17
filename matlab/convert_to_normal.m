function normal = convert_to_normal(normal, conf)
conf = repmat(conf, [1,1,3]);
normal = (normal - 0.5) * 2;
normal = normal .* conf;
end