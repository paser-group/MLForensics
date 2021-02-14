import unittest  

class TestCalc( unittest.TestCase ):
      
      def testHello(self):
          self.assertEqual( 3, 3, "Should be equal to 3" ) 
if __name__ == '__main__':
    unittest.main()   

