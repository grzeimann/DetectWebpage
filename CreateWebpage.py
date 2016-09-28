""" Create Webpage Class


1) Create a webpage based on 

.. moduleauthor:: Greg Zeimann <gregz@astro.as.utexas.edu>

"""        
        
class CreateWebpage(object):
    @classmethod
    def writeHeader(cls, f, title):
        """Write the header to webpage file ``f``

        Parameters
        ----------
        f : file-like object
            where to write to; must have a ``write`` method
        """
        s = []
        s.append('<!DOCTYPE HTML PUBLIC "-//W3C//DD HTML 4.01 Frameset//EN"')
        s.append('"http://www.w3.org/TR/html4/frameset.dtd">')
        s.append('<html>')
        s.append('<title> {:s}</title>'.format(title))
        s.append('<head>')
        s.append('<style type="text/css">')
        s.append('<!--')
        s.append('@import url(webpage_default.css);')
        s.append('-->')
        s.append('</style>')
        s.append('</head>')
        s.append('<h1 align=center> {:s} </h1>'.format(title))
        s.append('<script type="text/javascript" src="jquery.min.js"></script>')	    
        s.append(('<script type="text/javascript" '
                  'src="jquery.tablesorter.min.js"></script>'))
        f.write('\n'.join(s) + "\n")
    
    @classmethod
    def writeColumnNames(cls, f, columnnames, non_sortable_columns=None):
        """Write the column names to webpage file ``f``

        Parameters
        ----------
        f : file-like object
            where to write to; must have a ``write`` method
        """
        # Check if columnnames is a list
        if not isinstance(columnnames,list):
            columnnames = list(columnnames)
            
        s = []
        s.append('''    <script type="text/javascript" id="js">
    
            // Add ability to sort the table
            $(document).ready(function() {
                $.tablesorter.defaults.sortList = [[0,0]]; 
                $("table").tablesorter({
                        // pass the headers argument and assing a object
                        headers: {
                                // assign the secound column (we start counting zero)''')
        if non_sortable_columns:
            for ns in non_sortable_columns:
                s.append('''
                                %d: {
                                        sorter: false
                                },''' %(ns))
        s.append('''                }
                });        
            });
            </script>''')
        s.append('''    <table id="myTable" cellspacing="1" class="tablesorter"> 
            <thead>''')
        for col in columnnames:
            s.append('<th class="axe" height="35"> {:s} </th>'.format(col))
        s.append('</thead>')
        s.append('<tbody>')
        f.write('\n'.join(s) + "\n")
        f.flush()
        
    @classmethod
    def writeColumn(cls, f, dictionary):
        """Write rows to webpage file ``f``

        Parameters
        ----------
        f : file-like object
            where to write to; must have a ``write`` method
        """
        s = []
        key_list = ["number","link","table","image"]
        s.append('<tr class="axe">')
        for key, value in dictionary.iteritems():
            if key.lower().split('_')[0] not in key_list:
                print("[ERROR] Key, {:s}, not in key_list.")
                return None
            if key.lower().split('_')[0] == "number":
                if isinstance(value,int):
                    s.append(('<td  class="axe" ' 
                              'height="240">%d</td>' %(value)))
                if isinstance(value,float):
                    s.append(('<td  class="axe" ' 
                              'height="240">%0.3f</td>' %(value)))
            if key.lower().split('_')[0] == "link":
                if len(value)!=2:
                    print(('[ERROR] Value in Dictionary should include two '
                           'items: a string and a number.'))
                    return None
                s.append(('<td  class="axe" height="240"><a '
                          'href=%s>%d</a></td>' %(value[0],value[1])))
            if key.lower().split('_')[0] =="table":
                t=[]
                t.append('<td class="axe">')
                if isinstance(value,list):
                    for i in xrange(len(value)-1):
                        t.append('%s<br>' %(value[i]))
                    t.append('%s<p></td>' %(value[-1]))
                else:
                    t.append('%s<p></td>' %(value))
                s.append(''.join(t))
            if key.lower().split('_')[0]=="image":
                s.append('<td  class="axe" ><img align=middle '
                         'src=%s alt=%s></td>' %(value,value))    
        s.append('</tr>')
        f.write('\n'.join(s) + "\n")
        f.flush()
     
    @classmethod 
    def writeEnding(cls, f):
        """Write ending to webpage file ``f``

        Parameters
        ----------
        f : file-like object
            where to write to; must have a ``write`` method
        """
        s = []
        s.append('</tbody></table>')
        f.write('\n'.join(s) + "\n")
        f.flush()
        

    