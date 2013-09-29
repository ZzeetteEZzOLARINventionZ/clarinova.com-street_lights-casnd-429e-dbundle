'''

'''


from  databundles.bundle import BuildBundle
 

class Bundle(BuildBundle):
    ''' '''
 
    def __init__(self,directory=None):
        self.super_ = super(Bundle, self)
        self.super_.__init__(directory)

   
    def prepare(self):
        return super(Bundle, self).prepare()
        
        
    def build(self):
        self.copy_records()
        self.add_places()
        self.make_hdf()

        return True

    def copy_records(self):
        
        from datetime import date
        import ogr
        from databundles.datasets.geo import US
      
        place = US(self.library).place('SndSAN')
      
        streetlights = self.library.dep('streetlights').partition

        p = self.partitions.find_or_new_geo(table='streetlights')   
 
        lr = self.init_log_rate()

        with p.inserter(source_srs=streetlights.get_srs(), dest_srs=place.spsrs) as ins:
            for row in streetlights.query("""
            SELECT *, X(Transform(geometry,4326)) as lon,  Y(Transform(geometry,4326)) as lat,  
            AsText(geometry) as geometry
            FROM street_lights"""):
                ins.insert(dict(row))
                lr("Copying records")
             
        return True
 
    
 
    def add_places(self):      
        from databundles.geo.util import segment_points

        lr = self.init_log_rate(1000)

        places = self.library.dep('places').partition
        
        # This query is ordered from smallest area to largest, ans the inner loop ignored items that already have
        # a value set. This gets the smaller areas out of the way first, and reduces the points that have to be
        # checked for the larger areas. 

        table_name = 'streetlights'
        table = self.schema.table(table_name)
        p = self.partitions.find(table='streetlights', format='geo')   
  
        for area, query, is_in in segment_points(places, table_name='streetlights'):
            # The county is the largest area, and the default "remainder" so we don't have to processes it
            # It is really slow
            if area['code'] == 'SndSDO': continue

            with p.database.updater(table_name) as upd:
                for row in p.query(query):
                    if is_in(row['lon'], row['lat']):
                        lr("Add place: {} {} ({})".format(area['type'], area['name'], area['code']))

                        upd.update({'_OGC_FID': row['OGC_FID'],'_'+area['type'] : area['code'] })

    def make_hdf(self):
        
        import databundles.geo as dg
        from databundles.geo.analysisarea import get_analysis_area
        from osgeo.gdalconst import GDT_Float32
        import numpy as np

        scale = 10

        k = dg.DistanceKernel(21)

        k.matrix *= scale * 100 # Measured in cm: meters *  100 cm / m. 
        k.matrix = np.array(k.matrix, dtype=int) # Intify
        
        p = self.partitions.find(table='streetlights', format='geo')
        
        if not p:
            raise Exception("Didn't find partition for streetlights")
        
        raster = self.partitions.find_or_new_hdf(table='distance')        
        
        for cityrow in p.query("SELECT count(*) AS count, city FROM streetlights group BY city"):

            city = cityrow['city']    
            
            if not city or city == '-':
                continue;
            
            if cityrow['count'] < 100:
                continue # Most of the cityies have a, probably spurious, small number of lamps
            
            self.log("Making HDF for city {}".format(city))
            
            lr = self.init_log_rate()
            
            aa = get_analysis_area(self.library, place=city, scale=scale)
            trans = aa.get_translator()
            a = aa.new_array(dtype = np.int16)

            a = a + 12000 # Set the baseline to a max at 120 meters. 

            for row in p.query("""SELECT * FROM streetlights WHERE city = ?""", city):
                k.apply_min(a, trans(row['lon'], row['lat']))  
                lr("Distance raster point for {}".format(city))

            raster.database.put_geo(city, a, aa)
            raster.database.flush()

            aa.write_geotiff( self.filesystem.path('extracts', city+"-dist"), a[:])
            

    def make_distance_map(self):
        import numpy as np
        import numpy.ma as ma
        import databundles.geo as dg
            
        np.set_printoptions(precision=1, linewidth=240, threshold=10000, suppress = True)
            
        k = dg.DistanceKernel(21)
        k.matrix *= 10 * 100 # 10M by 100 cm / m. 
        k.matrix = np.array(k.matrix, dtype=int)
        print ma.filled(k.matrix,0) # printoptions precision doesn't work on masked arrays
        
        a = np.arange(20*20, dtype = float).reshape(20,20)


        k.apply_min(a, dg.Point(10,10))
        
        print
        print a
        
        k.apply_min(a, dg.Point(12,12))
        
        print
        print a

    def extract_colormaps(self, data):
        import databundles.geo.colormap as cm
        import numpy as np

        raster = self.partitions.find(table='distance')     
        a,_ = raster.database.get_geo('SndSAN')
     
        a1 = np.sort(a[...].ravel()) #.astype(np.float32)
     
        #a1 /= np.max(a1)
     
        cmap =  cm.get_colormap(data['map_name'],9, reverse=bool(data['reversed']))
        
        path = data['path']

        cm.write_colormap(path, a1, cmap,  break_scheme = data['break'])
   
        return path
 
 
    def stats(self, data):
        """Produce stats for count of lamps and densities. """
        from csv import writer
        
        p = self.partitions.find_or_new(table='streetlights')   
 
        p.database.attach(neighborhoods,'nb')
 
        name = data['name']
        
        # The areas are in square feet. WTF?
        feetperm = 3.28084
        feetperkm = feetperm * 1000
        
        
        with open(self.filesystem.path('extracts',name), 'wb') as f:
            writer = writer(f)
            writer.writerow(['count', 'neighborhood','area-sqft','area-sqm','area-sqkm', 'density-sqkm',])
            
            for row in p.database.query("""
            SELECT count(streetlights_id) as count, objectid, cpname, shape_area
            FROM streetlights, {nb}.communities
            WHERE streetlights.neighborhood_id = {nb}.communities.objectid
            GROUP BY {nb}.communities.objectid
            """):
                
                n = float(row['count'])
                area = float(row['shape_area'])
                
                writer.writerow([ 
                n,  
                row['cpname'].title(),
                area,
                area / (feetperm * feetperm),
                area / (feetperkm * feetperkm),
                n / (area / (feetperkm * feetperkm)) 
                ])


    def extract_image(self, data):
        
        import databundles.geo as dg
        from databundles.geo.analysisarea import get_analysis_area

        place = 'SndSAN'
        
        raster = self.partitions.find(table='distance', format='hdf')        
        a, aa = raster.database.get_geo(place)

        file_name = self.filesystem.path('extracts', data['name'])
        
        aa.write_geotiff(file_name, a[:])
        
        return file_name

    def extract_shapefiles(self, data):
        import pprint
        from databundles.geo.sfschema import TableShapefile

        name = data['name']
        fpath = self.filesystem.path('extracts', name)

        streetlights = self.partitions.find(table='streetlights', type='geo')

        with TableShapefile(self, fpath, 'streetlights', source_srs = 4326, name = data['layer_name']) as tsf:      
            self.log("Extracting {}".format(name))
            lr = self.init_log_rate()

            for row in streetlights.query('SELECT *, AsText(geometry) as geometry from streetlights'):
                row = dict(row)

                if row['lat'] > 0: # Not strictly correct, but OK for San Diego
                    lr("Extract feature for {} ".format(name))
                    tsf.add_feature(row)

        return fpath

    def convert_partition_from_evari(self):
        
        from lxml import etree
        from StringIO import StringIO
        from pprint import pprint
        import re

        lr = self.init_log_rate(message='Convert lights partition: ')
        
        dest_p = self.partitions.find_or_new(table='lights')
           
        dest_p.database.connection.execute("DELETE FROM lights")
           
        with dest_p.database.inserter() as ins:
            for row in p.query("""
                SELECT 
                    description, 
                    X(Transform(geometry, 4326)) AS lon, 
                    Y(Transform(geometry, 4326)) AS lat 
                FROM lights_g"""):

                    lr()
                    
                    r = etree.HTML(row['description'])
    
                    table = r.find('.//table').find('.//table')
    
                    trs = iter(table)
    
                    d = {}
                    for tr in trs:
                        values = [col.text for col in tr]
                        d[values[0].lower().replace(' ','_')] = values[1]
                        
                
                    if ( 'not part' in d['conversion_status'] ):
                        status = "not in project"
                    elif  'not been converted' in d['conversion_status'] :
                        status = 'not converted'
                    elif  'has been converted' in d['conversion_status']:
                        status = 'converted'
                    elif 'No Light Here' in d['conversion_status']:
                        status = 'no light'
                    else:
                        status = 'other'
                        
                    try: old_wattage = None if not d['existing_field_wattage'] else re.match('(\d+)', d['existing_field_wattage']).group(1)
                    except: old_wattage = None
                    
                    try: new_wattage = None if not d['new_wattage'] else re.match('(\d+)', d['new_wattage']).group(1)
                    except: new_wattage = None
                    
                    new_type = d['new_type']
                    
                    if d['existing_field_type'] and 'High' in d['existing_field_type']:
                        old_type = 'HPS'
                    elif d['existing_field_type'] and 'Low' in d['existing_field_type']:
                        old_type = 'LPS'
                    else:
                        old_type = None
                    
                    if d['new_type'] and "IND" in d['new_type']:
                        new_tpye = 'IND'
                    else:
                        new_type = None
                    
                    r = {
                         'status':status, 
                         'old_wattage':old_wattage, 
                         'new_wattage':new_wattage, 
                         'old_type':old_type,
                         'new_type':new_type, 
                         'lat':row['lat'],
                         'lon':row['lon']
                        }
                    
                    ins.insert(r)


    def evari_extract_image(self, data):
        
        import databundles.geo as dg
        from databundles.geo.analysisarea import get_analysis_area
        from osgeo.gdalconst import GDT_Float32
        
        aa = get_analysis_area(self.library, geoid='CG0666000')
        trans = aa.get_translator()

        a_old = aa.new_array()
        a_new = aa.new_masked_array()
        a_nip = aa.new_array()
        a_total = aa.new_masked_array()
        
        k = dg.GaussianKernel(21,7)
        
        p = self.partitions.find(table='lights')
        
        for row in p.query("""SELECT * FROM lights """):
       
            p = trans(row['lon'], row['lat'])
       
            status = row['status']
       
            if status == 'not in project':
                k.apply_add(a_nip, p)  
            elif status == 'converted':
                k.apply_add(a_new, p)
                k.apply_add(a_total, p) 
            elif status == 'not converted':
                k.apply_add(a_old, p) 
                k.apply_add(a_total, p)  
  
            
        def fnp(prefix):
             return self.filesystem.path('extracts','{}-{}'.format(prefix,data['name']))

        aa.write_geotiff(fnp('old'), a_old, data_type=GDT_Float32)
        aa.write_geotiff(fnp('new'), a_new, data_type=GDT_Float32)
        aa.write_geotiff(fnp('nip'), a_nip, data_type=GDT_Float32)
        aa.write_geotiff(fnp('total'), a_total, data_type=GDT_Float32)
        
        pct = 1 - (a_new / a_total)
        
        aa.write_geotiff(fnp('pct'), pct, data_type=GDT_Float32)
        
        
        return fnp('total')
 

    
import sys

if __name__ == '__main__':
    import databundles.run
      
    databundles.run.run(sys.argv[1:], Bundle)
     
    
    